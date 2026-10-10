"""
run_experiments.py

sotsuron_plan2.tex 5.2：検知評価スイープを自動で回すオーケストレーションスクリプト。

- 更新ベクトルレベルの異常検知を，Non-IID度（Dirichlet alpha）× 攻撃 の2軸で評価する。
  全組み合わせについて `flwr run` をサブプロセスとして実行する。
- 【設定値はこのファイルに書かない】条件はすべて pyproject.toml から読む。
    [tool.flwr.app.config] : 全条件で共通の値（クライアント数・ラウンド数・学習率など）
    [tool.sweep]           : 変えながら回す軸（集約手法・alpha・攻撃・ALIE の z・trial）
  このスクリプトは [tool.sweep] の軸に対応する項目だけを --run-config で上書きする。
- 攻撃なし（none）は benign_spread の基準値を取るための条件で，悪性割合を 0.0 にして回す。
- 各実験は <output-root>/<条件名>/ に出力先を分ける（final_model.pt が実験間で
  上書きされないようにする）。
- 同一条件をtrial回数分繰り返す際、trial番号はデータ分割と悪性クライアント選出の両方に
  連動させる（malicious-seed にも同じ値を渡す）。これにより「どのデータ分割か」「どの
  クライアントが悪性か」の両方がtrialごとに変わり、より代表性のある平均が得られる。

【完了判定・再開について】
`flwr run` はサブプロセスとしては例外発生時でも終了コード0を返すことがある
（Flower内部でエラーがログに出るだけでプロセス自体は正常終了扱いになるため）。
そのため成功判定は終了コードではなく、期待される出力ファイル
（<output-root>/<条件名>/final_model.pt）が実際に生成されたかどうかで行う。
このファイルが既に存在する条件は「完了済み」としてスキップするため、
スクリプトを中断して再実行すれば未完了分だけ再開できる。

【失敗の記録について】
成功しなかった条件は <output-root>/failures.csv に記録し、スイープ全体は止めずに次へ進む。

実行方法:
    python3 run_experiments.py              # 全条件を実行（未完了分のみ）
    python3 run_experiments.py --dry-run    # 実行せず，条件と flwr run のコマンドを表示するだけ
"""

# 【このファイルの読み方】
#   実験を「まとめて回す」ための補助スクリプト。研究の中身（攻撃・検知・学習）は何も入っていない。
#   やっていることは，人が手で次のコマンドを54回打つ代わりをしているだけである。
#       flwr run . local --run-config "条件ごとに違う設定" --stream
#   連合学習そのものは，このコマンドで起動される fl_detection/ の中のコードが行う。
#
#   全体の流れ（一番下の main() が，上から順に実行する）
#       1. pyproject.toml から，共通の設定と，変えながら回す軸を読む            load_settings
#       2. 軸の全組み合わせを並べて，条件の一覧（54件）を作る                    detection_conditions
#       3. Flower のコマンドが使えるか確かめ，クライアントの台数を設定する        check_flwr_available / ensure_simulation_config
#       4. 条件を1つずつ順に実行する                                              run_one_detection
#            - すでに終わっている条件は飛ばす                                     is_done
#            - flwr run のコマンドを組み立てて実行する                            build_command / build_run_config
#            - 終わったら，成功したか失敗したかを確かめる（失敗は記録して次へ）   record_failure
#       5. 完了・スキップ・失敗の件数を表示する
#
#   使い方
#       python run_experiments.py --dry-run   実行はせず，条件の一覧とコマンドを表示するだけ
#       python run_experiments.py             実行する（途中で止めても，もう一度実行すれば続きから回る）
#
# 【import の説明】（どれも Python 標準の道具）
import argparse                 # コマンドの後ろに付ける指定（--dry-run など）を読み取る
import csv                      # CSV ファイルの読み書き
import os                       # ファイルやフォルダ，環境変数を扱う
import shutil                   # コマンドがどこにあるかを探す（shutil.which）
import subprocess               # ほかのコマンド（ここでは flwr）を，Python の中から実行する
import sys                      # 今動いている Python の場所など（sys.executable）
from datetime import datetime   # 日付と時刻

# pyproject.toml（TOML 形式）を読むための道具。Python 3.11 以降は標準の tomllib を使い，
# それより古い場合は，同じ働きをする tomli を tomllib という名前で読み込む（as で別名を付けている）。
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 では flwr が依存している tomli を使う
    import tomli as tomllib


# ============================================================
# Flower のコマンドを呼ぶための準備
# ============================================================
# 【この部分の説明】
#   flwr コマンドを Python の中から実行するための下準備をする関数たち。
#
#   環境変数とは
#     コマンドを実行するときに，周りから渡される設定の集まり。その1つである PATH は
#     「コマンドを探すフォルダの一覧」で，flwr と打ったときにどこの flwr が動くかは PATH で決まる。
#
#   _subprocess_env()       : flwr を呼ぶときに渡す環境変数を作る。
#       os.environ.copy()   : 今の環境変数の写しを作る（写しを書き換えるので，もとには影響しない）。
#       os.path.dirname(sys.executable) : 今動いている Python が入っているフォルダ。
#                             仮想環境の Python で実行していれば，flwr もこのフォルダに入っている。
#       os.pathsep          : PATH の中でフォルダを区切る文字（Mac では : ）。
#   check_flwr_available()  : flwr と flower-superlink が見つかるかを確かめ，無ければその場で止める。
#       shutil.which(名前, path=...) : そのコマンドの場所を探す。見つからなければ None を返す。
#       raise SystemExit(メッセージ)  : メッセージを表示して，スクリプト全体を終了する。
#   check_gpu_settings()    : GPU の設定がこのマシンと食い違っていないかを確かめ，食い違っていれば止める。
#   _flwr_bin() / _flwr_cmd_prefix() : 実行する flwr の場所を決める。コマンドの先頭部分として使う。
#   ensure_simulation_config()       : 起動するクライアントの台数と，各クライアントへの GPU の割り当てを，Flower に設定する。
#       この台数が pyproject.toml の num-clients と合っていないと，実験は最初のラウンドで止まる
#       （fl_detection/robust_strategy.py の台数の照合）。
#       subprocess.run(コマンドのリスト, ...) : コマンドを実行し，終わるまで待つ。
#           capture_output=True : コマンドが画面に出す文字を，画面に出さずに受け取る。
#           proc.returncode     : コマンドの終了コード。0 が成功，それ以外は失敗。
#       *_flwr_cmd_prefix() の * : リストの中身を，その場に展開して並べる書き方。
def _subprocess_env() -> dict:
    """flwr を呼ぶときの環境変数を作る。

    - PYTHONUTF8=1 : flwr CLIの絵文字混じり出力がWindowsのcp932コンソールで落ちるのを防ぐ。
    - PATH の先頭に「今の Python と同じフォルダ」を足す：
        `flwr run` は内部で flower-superlink という別のコマンドを PATH から探して起動する。
        venv を activate せずに `.venv/bin/python run_experiments.py` と実行すると PATH に
        venv が入っておらず，全条件が「flower-superlink が見つからない」で失敗してしまうため。
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    return env


def check_flwr_available() -> None:
    """Flower のコマンドが見つからなければ，1条件も回さないうちに止める。"""
    path = _subprocess_env()["PATH"]
    missing = [c for c in ("flwr", "flower-superlink") if shutil.which(c, path=path) is None]
    if missing:
        raise SystemExit(
            f"Flower のコマンドが見つかりません: {missing}\n"
            "仮想環境の Python で実行してください（例: .venv/bin/python run_experiments.py）。"
        )


def check_gpu_settings(device: str, client_num_gpus: float) -> None:
    """GPU の設定が，このマシンと食い違っていないかを確かめる。食い違っていれば，1条件も回さずに止める。

    NVIDIA の GPU（cuda）は，サーバからは常に見えるが，クライアントからは
    [tool.sweep] の client-num-gpus で割り当てたときだけ見える（Flower がクライアントを別プロセスで動かすため）。
    そのため，設定が食い違うと「GPU で回しているつもりが，クライアントは CPU で学習している」
    という状態が，エラーなしで起きる。それを防ぐ。
    """
    import torch  # このスクリプトでは，ここでしか使わないので，関数の中で読み込む

    has_cuda = torch.cuda.is_available()
    if client_num_gpus > 0 and not has_cuda:
        raise SystemExit(
            f"client-num-gpus = {client_num_gpus} ですが，このマシンでは NVIDIA の GPU（cuda）が使えません。\n"
            "GPU の無いマシン（Mac を含む）では，クライアントが起動できずに止まったままになります。\n"
            "pyproject.toml の [tool.sweep] の client-num-gpus を 0.0 にしてください。\n"
            "GPU があるはずのマシンでこの表示が出る場合は，PyTorch が GPU 対応の版かを確かめてください"
            "（python -c \"import torch; print(torch.cuda.is_available())\" が True になること）。"
        )
    if has_cuda and client_num_gpus == 0 and device != "cpu":
        raise SystemExit(
            f"NVIDIA の GPU が使えるマシンですが，client-num-gpus = 0.0 のため，クライアントは CPU で学習します"
            f"（device = \"{device}\"）。\n"
            "GPU で回すなら，pyproject.toml の [tool.sweep] の client-num-gpus を 0 より大きく（例: 0.2）してください。\n"
            "CPU で回すなら，[tool.flwr.app.config] の device を \"cpu\" にしてください。"
        )


def _flwr_bin() -> str:
    """
    実行する flwr コマンドのパスを決める。
    venv を activate せずに `python3 run_experiments.py` を実行しても動くよう、
    まず現在の Python と同じディレクトリの flwr を探し、無ければ PATH 上の flwr、
    それも無ければ `python -m flwr` にフォールバックする。
    """
    cand = os.path.join(os.path.dirname(sys.executable), "flwr")
    if os.path.exists(cand):
        return cand
    found = shutil.which("flwr")
    if found:
        return found
    return None  # 呼び出し側で [sys.executable, "-m", "flwr"] に切り替える


def _flwr_cmd_prefix() -> list:
    b = _flwr_bin()
    return [b] if b else [sys.executable, "-m", "flwr"]


def ensure_simulation_config(federation: str, num_clients: int, client_num_gpus: float) -> None:
    """SuperLink接続（federation）のシミュレーション設定（クライアント数）を確定させる。

    新しいflwr（SuperLink接続方式）では `flwr run --federation-config` は非推奨で無視される
    ため、`flwr federation simulation-config` で事前に num-supernodes を設定しておく必要が
    ある（未設定時のデフォルトは2クライアントで、実験条件と食い違ってしまう）。
    """
    # 第1位置引数はfederation ID（クラウド用、`@account/name`形式）、第2位置引数が
    # SuperLink接続名。ローカルシミュレーションではfederation IDを使わないため空文字を渡す。
    cmd = [
        *_flwr_cmd_prefix(), "federation", "simulation-config", "", federation,
        "--num-supernodes", str(num_clients),
        # 各クライアントに割り当てる NVIDIA GPU の割合。0 なら割り当てない（CPU，または Mac の GPU で動く）。
        # Flower はクライアントを別々のプロセスで動かしており，ここで割り当てないと，
        # クライアントからは NVIDIA の GPU が見えない。
        "--client-resources-num-gpus", str(client_num_gpus),
    ]
    print(f"[SETUP] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", env=_subprocess_env()
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"federation '{federation}' の simulation-config 設定に失敗しました:\n"
            f"{(proc.stdout or '') + (proc.stderr or '')}"
        )


# ============================================================
# 条件の一覧を作る
# ============================================================
# 【この部分の説明】
#   FAILURES_CSV_NAME : 失敗した条件を記録するファイルの名前。
#   PYPROJECT_PATH    : pyproject.toml の場所。このスクリプトと同じフォルダにあるものを指す。
#       __file__                 : このスクリプト自身のファイル名。
#       os.path.abspath(...)     : それを，フォルダの位置まで含んだ完全な形（絶対パス）にする。
#       os.path.dirname(...)     : そこから，フォルダの部分だけを取り出す。
#
#   load_settings()
#       pyproject.toml を読み，（共通の設定, 変えながら回す軸）の2つを返す。
#       open(パス, "rb")         : ファイルを読み込み用に開く（"rb" は，文字ではなくデータのまま読む指定。tomllib の決まり）。
#       tomllib.load(f)["tool"]  : ファイル全体を入れ子の辞書として読み，その中の [tool.〜] の部分を取り出す。
#       tool["flwr"]["app"]["config"] : [tool.flwr.app.config] の欄。 tool["sweep"] : [tool.sweep] の欄。
#
#   detection_conditions(base, sweep)
#       軸の全組み合わせを，1件ずつ辞書にして取り出せるようにする。for が5重になっているのは，
#       集約手法 × alpha × 攻撃 ×（ALIE のときだけ z）× trial のすべての組み合わせを順に作るため。
#           1 × 3 ×（攻撃なし 1 + 符号反転 1 + ALIE の z 4 通り）× 3 = 54 件
#       ratio = 0.0 if attack == "none" else ... : 攻撃なしのときは悪性割合を 0 にする。
#       for z in (sweep["alie-zs"] if attack == "alie" else [None]):
#           ALIE のときは z の一覧を順に試し，それ以外の攻撃では「z なし」（None）を1回だけ回す。
#       yield {...}
#           return に似ているが，値を1つ渡したあと，関数を終わらせずに続きを待つ。
#           for で回すと，条件が1件ずつ順に出てくる（こういう関数をジェネレータと呼ぶ）。
#
#   detection_condition_name(p, base)
#       条件に名前を付ける。これがそのまま，出力フォルダの名前になる。
#           例: fedavg_dirichlet_a0.1_alie_z0.5_ratio0.2_trial2
FAILURES_CSV_NAME = "failures.csv"
PYPROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml")


def load_settings() -> tuple:
    """pyproject.toml から (共通設定, スイープの軸) を読む。

    共通設定は [tool.flwr.app.config]，スイープの軸は [tool.sweep]。
    設定値の置き場所を pyproject.toml の1か所にするため，このスクリプトには値を書かない。
    """
    with open(PYPROJECT_PATH, "rb") as f:
        tool = tomllib.load(f)["tool"]
    return tool["flwr"]["app"]["config"], tool["sweep"]


def detection_conditions(base: dict, sweep: dict):
    """検知評価スイープの条件を dict で列挙する。"""
    for aggregation in sweep["aggregation-methods"]:
        for alpha in sweep["dirichlet-alphas"]:
            for attack in sweep["attack-types"]:
                # 攻撃なしは悪性0人で回す（server_app.py も attack-type="none" なら自動で0人にするが，
                # 条件名と --run-config に実際の値が出るよう，ここでも明示する）
                ratio = 0.0 if attack == "none" else float(base["malicious-ratio"])
                for z in (sweep["alie-zs"] if attack == "alie" else [None]):
                    for trial in sweep["trials"]:
                        yield {
                            "aggregation": aggregation,
                            "alpha": float(alpha),
                            "attack": attack,
                            "z": None if z is None else float(z),
                            "ratio": ratio,
                            "trial": int(trial),
                        }


def detection_condition_name(p: dict, base: dict) -> str:
    attack = p["attack"] if p["z"] is None else f"{p['attack']}_z{p['z']}"
    return (
        f"{p['aggregation']}_{base['data-split']}_a{p['alpha']}_{attack}"
        f"_ratio{p['ratio']}_trial{p['trial']}"
    )


# ============================================================
# 1条件を実行する
# ============================================================
# 【この部分の説明】
#   build_run_config(p, output_dir)
#       flwr run の --run-config に渡す文字列を作る。pyproject.toml の値のうち，
#       この条件で上書きしたい項目だけを「名前=値」の形で並べる。
#           例: aggregation-method="fedavg" dirichlet-alpha=0.1 ... attack-type="alie" alie-z=0.5 output-dir="..."
#       ここに無い項目（クライアント数・ラウンド数・学習率など）は，pyproject.toml の値のまま動く。
#       f'...="{値}"' : 文字列の値は " " で囲む決まり（TOML の書き方）。外側を ' ' にして，中で " を使えるようにしている。
#       " ".join(items) : リストの中身を，空白でつないで1つの文字列にする。
#
#   build_command(p, output_dir, federation)
#       実行するコマンド全体を，単語ごとのリストとして組み立てる。
#           [flwr の場所, "run", ".", 接続名, "--run-config", 上の文字列, "--stream"]
#       "." は「今いるフォルダのアプリ」，--stream は「終わるまで待って，出力を流す」の指定。
#
#   run_one_detection(p, base, output_root, federation)
#       1条件を実行して，結果を "done"（完了）/ "skipped"（飛ばした）/ "failed"（失敗）の文字列で返す。
#         1. すでに final_model.pt があれば，実行せずに飛ばす。
#            これがあるので，スイープを途中で止めても，もう一度実行すれば未完了の条件から続けられる。
#         2. 出力フォルダを作り，コマンドを実行して，終わるまで待つ（ここで数分かかる）。
#         3. 終わったあと，final_model.pt ができていれば成功，なければ失敗とする。
#            flwr run は，中でエラーが起きても「成功」の終了コードを返すことがあるので，
#            終了コードではなく，最後に作られるはずのファイルがあるかどうかで判断している。
#         4. 失敗なら，出力の最後の30行を画面に出し，失敗の一覧（failures.csv）に記録する。
#            スイープ全体は止めずに，次の条件へ進む。
#       (proc.stdout or "") : 出力が None だった場合は，空の文字列として扱う。
#       .splitlines()[-30:] : 行ごとに分けて，最後の30行だけを取り出す。
#
#   is_done(output_dir)     : そのフォルダに final_model.pt があるかどうか（True / False）。
#   record_failure(...)     : 失敗した条件の名前・コマンド・出力の末尾を，failures.csv に1行追記する。
def build_run_config(p: dict, output_dir: str) -> str:
    """1条件ぶんの --run-config 文字列を組み立てる。

    渡すのは [tool.sweep] の軸に対応する項目と出力先だけ。
    それ以外（クライアント数・ラウンド数・学習率など）は pyproject.toml の
    [tool.flwr.app.config] の値がそのまま使われる。
    """
    # --run-config はTOML文字列として解釈されるため、Windowsのバックスラッシュ区切り
    # パスをそのまま埋め込むと "\f" 等がエスケープシーケンスとして解釈され、
    # パスが壊れてしまう（例: "\fedavg" -> フォームフィード文字+"edavg"）。
    # スラッシュ区切りに変換して渡す（Windows/Pythonのos.path系はスラッシュも受理する）。
    output_dir_for_config = output_dir.replace(os.sep, "/")

    items = [
        f'aggregation-method="{p["aggregation"]}"',
        f'dirichlet-alpha={p["alpha"]}',
        f'malicious-ratio={p["ratio"]}',
        f'malicious-seed={p["trial"]}',
        f'trial={p["trial"]}',
        f'attack-type="{p["attack"]}"',
    ]
    if p["z"] is not None:
        items.append(f'alie-z={p["z"]}')
    items.append(f'output-dir="{output_dir_for_config}"')
    return " ".join(items)


def build_command(p: dict, output_dir: str, federation: str) -> list:
    # 新しいflwr（SuperLink接続方式）では --federation-config は非推奨で無視される。
    # クライアント数は main() 側であらかじめ ensure_simulation_config() で設定しておく。
    return [
        *_flwr_cmd_prefix(), "run", ".", federation,
        "--run-config", build_run_config(p, output_dir),
        "--stream",  # --streamを付けないと flwr run は完了を待たずに即座に返ってしまう
    ]


def run_one_detection(p: dict, base: dict, output_root: str, federation: str) -> str:
    """検知評価スイープの1条件を実行する。戻り値は "done" / "skipped" / "failed"。"""
    name = detection_condition_name(p, base)
    output_dir = os.path.join(output_root, name)

    if is_done(output_dir):
        print(f"[SKIP] {name}（{output_dir}/final_model.pt が既に存在＝完了済み）")
        return "skipped"

    os.makedirs(output_dir, exist_ok=True)
    cmd = build_command(p, output_dir, federation)

    print(f"[RUN] {name}")
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", env=_subprocess_env()
    )

    if is_done(output_dir):
        print(f"[DONE] {name}")
        return "done"

    # final_model.pt が無ければ失敗とみなす（終了コードは当てにできないため）
    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    output_tail = "\n".join(combined_output.strip().splitlines()[-30:])
    print(f"[FAILED] {name}")
    print(output_tail)
    record_failure(name, cmd, output_tail, experiments_dir=output_root)
    return "failed"


def is_done(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "final_model.pt"))


def record_failure(name: str, cmd: list, output_tail: str, experiments_dir: str) -> None:
    os.makedirs(experiments_dir, exist_ok=True)
    failures_csv = os.path.join(experiments_dir, FAILURES_CSV_NAME)
    file_exists = os.path.exists(failures_csv)
    with open(failures_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "condition", "command", "output_tail"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name,
            " ".join(cmd),
            output_tail,
        ])


# ============================================================
# 全体の進行
# ============================================================
# 【main の説明】
#   スクリプトを実行したときに動く本体。冒頭の「全体の流れ」を，上から順に行う。
#
#   コマンドの後ろに付ける指定を読む部分
#     parser = argparse.ArgumentParser(...)  : 指定を読み取る係を作る。
#     parser.add_argument("--dry-run", action="store_true", help=...)
#         : 受け付ける指定を1つ登録する。action="store_true" は「付いていれば True，無ければ False」。
#           help は，python run_experiments.py --help と打ったときに表示される説明。
#     args = parser.parse_args()             : 実際に付けられた指定を読む。args.dry_run のように取り出せる
#                                              （--dry-run のハイフンは，名前の中ではアンダースコアになる）。
#
#   条件を回す部分
#     conditions = list(detection_conditions(base, sweep)) : 条件を全部取り出して，リストにする。
#     for i, p in enumerate(conditions, start=1): : 条件を1件ずつ取り出す。i は 1 から始まる通し番号。
#     print(f"[{i}/{total}]", end=" ")        : 「[3/54]」のように進み具合を表示する。end=" " は，改行せずに続ける指定。
#     counts[run_one_detection(...)] += 1     : 返ってきた結果（"done" など）の件数を，1つ増やす。
#
#   if args.dry_run: ... return               : --dry-run のときは，条件とコマンドを表示するだけで終わる。
#                                               Flower は起動せず，何も書き込まない。
def main() -> None:
    parser = argparse.ArgumentParser(description="sotsuron_plan2.tex 5.2：検知評価スイープの自動実行")
    parser.add_argument(
        "--detection",
        action="store_true",
        help="（互換用・指定不要）このスクリプトは検知評価スイープだけを実行する。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せず，条件名と flwr run のコマンドを表示するだけにする（pyproject.toml の確認用）。",
    )
    parser.add_argument("--federation", default="local", help="flwr run に渡すSuperLink接続名（~/.flwr/config.tomlに定義されたもの）")
    parser.add_argument(
        "--aggregations",
        default=None,
        help="実行対象のaggregation_methodをカンマ区切りで指定（未指定時は [tool.sweep] の全手法）。"
        "例：--aggregations fedavg で再実験など一部手法のみ再実行したい場合に使う。",
    )
    args = parser.parse_args()

    base, sweep = load_settings()
    output_root = sweep["output-root"]
    num_clients = int(base["num-clients"])
    client_num_gpus = float(sweep["client-num-gpus"])

    conditions = list(detection_conditions(base, sweep))
    if args.aggregations is not None:
        selected_aggregations = {a.strip() for a in args.aggregations.split(",")}
        conditions = [c for c in conditions if c["aggregation"] in selected_aggregations]
    total = len(conditions)

    print(f"=== 検知評価スイープ 条件数: {total}（出力先: {output_root}/） ===")
    print(
        f"共通設定: {num_clients}クライアント・{base['num-server-rounds']}ラウンド・"
        f"学習率{base['learning-rate']}・data-split={base['data-split']}・"
        f"device={base['device']}（クライアントへの GPU 割り当て: {client_num_gpus}）\n"
    )

    if args.dry_run:
        for i, p in enumerate(conditions, start=1):
            name = detection_condition_name(p, base)
            cmd = build_command(p, os.path.join(output_root, name), args.federation)
            print(f"[{i}/{total}] {name}")
            print(f"    {' '.join(cmd)}")
        return

    check_flwr_available()
    check_gpu_settings(base["device"], client_num_gpus)
    ensure_simulation_config(args.federation, num_clients, client_num_gpus)

    counts = {"done": 0, "skipped": 0, "failed": 0}
    for i, p in enumerate(conditions, start=1):
        print(f"[{i}/{total}]", end=" ")
        counts[run_one_detection(p, base, output_root, args.federation)] += 1

    print("\n=== 検知評価スイープ終了 ===")
    print(f"完了: {counts['done']} / スキップ（完了済み）: {counts['skipped']} / "
          f"失敗: {counts['failed']} / 合計: {total}")
    if counts["failed"] > 0:
        print(f"失敗した条件は {os.path.join(output_root, FAILURES_CSV_NAME)} を参照してください。")


# このファイルを「python run_experiments.py」と直接実行したときだけ，main() を呼ぶ。
# ほかのファイルから import されたとき（たとえばテスト）は，関数が使えるようになるだけで，実験は始まらない。
#   __name__ は，直接実行されたときだけ "__main__" という値になる，Python が用意している変数。
if __name__ == "__main__":
    main()
