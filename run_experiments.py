"""
run_experiments.py

実験計画書 7.6 対応：全実験条件を自動で回すオーケストレーションスクリプト。

- aggregation / malicious_ratio / data_split / trial の一覧を定義し、全組み合わせ
  （5 x 4 x 2 x 20 = 800通り）について `flwr run` をサブプロセスとして実行する。
  trial数は Fang et al. (2020) の "trial"（同一条件を50 trials繰り返して平均する）に
  合わせた表記・考え方だが，計算コストの都合で20 trialsに設定している。
- 実験計画書5.2の比較条件（集約手法・悪性クライアント割合・データ分布）に沿っており、
  attack-type は本研究の主眼である "adaptive"（Adaptive Model Poisoning Attack）に固定する
  （5.2の比較条件に攻撃種別は含まれていないため）。
- 各実験は experiments/<条件名>/ に出力先を分ける（final_model.pt / fixed_eval が
  実験間で上書きされないようにする、実験計画書7.6「各実験ごとに出力ディレクトリを分ける」対応）。
- 同一条件をtrial回数分繰り返す際、trial番号はデータ分割と悪性クライアント選出の両方に
  連動させる（malicious-seed にも同じ値を渡す）。これにより「どのデータ分割か」「どの
  クライアントが悪性か」の両方がtrialごとに変わり、より代表性のある平均が得られる。

【完了判定・再開について】
`flwr run` はサブプロセスとしては例外発生時でも終了コード0を返すことがある
（Flower内部でエラーがログに出るだけでプロセス自体は正常終了扱いになるため）。
そのため成功判定は終了コードではなく、期待される出力ファイル
（experiments/<条件名>/final_model.pt）が実際に生成されたかどうかで行う。
このファイルが既に存在する条件は「完了済み」としてスキップするため、
スクリプトを中断して再実行すれば未完了分だけ再開できる。

【失敗の記録について】
成功しなかった条件は experiments/failures.csv に記録し、スイープ全体は止めずに次へ進む。
例：Bulyanは悪性クライアント割合30%・クライアント数100だと n >= 4f+3 を満たせず、
robust_strategy.py が意図的にValueErrorを送出する（既知のBulyanの限界であり、バグではない）。

実行方法:
    python3 run_experiments.py            # 全800条件を実行（未完了分のみ）
    python3 run_experiments.py --quick     # 動作確認用の小規模設定で試す
    python3 run_experiments.py --phase1    # report/mechanism_analysis_plan.tex のPhase 1を実行
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime


def _utf8_env() -> dict:
    """flwr CLIの絵文字混じり出力がWindowsのcp932コンソールで落ちるのを防ぐ。"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


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


def ensure_simulation_config(federation: str, num_clients: int) -> None:
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
    ]
    print(f"[SETUP] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", env=_utf8_env()
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"federation '{federation}' の simulation-config 設定に失敗しました:\n"
            f"{(proc.stdout or '') + (proc.stderr or '')}"
        )


AGGREGATION_METHODS = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
MALICIOUS_RATIOS = [0.0, 0.1, 0.2, 0.3]
DATA_SPLITS = ["iid", "non_iid"]
TRIALS = list(range(1, 21))  # Fang et al. (2020)は50 trials。本実験は計算コストの都合で20 trials

NUM_CLIENTS = 100
NUM_SERVER_ROUNDS = 20  # 予備実験で決定した値に差し替えること（実験計画書6章）
ATTACK_TYPE = "adaptive"  # 本研究の主眼（実験計画書4.4）。5.2の比較条件に攻撃種別は含まれない

EXPERIMENTS_DIR = "experiments"
FAILURES_CSV_NAME = "failures.csv"

# ============================================================
# report/mechanism_analysis_plan.tex：Phase 1（探索）の条件定義
#
# 全条件を対象に、既存4段階のMalicious Ratioに加えて、臨界点探索のため
# 手法ごとにRatioを細分化する（trial数は全条件TRIALSのまま）。
# 出力先は experiments/ とは別の experiments_phase1/ に分ける。
# 既存 experiments/ の final_model.pt には診断ログ（logs/diagnostics.csv）が
# 含まれていない（本セッションでロギングを追加する前の実行のため）ため、
# 同じ出力先を使い回すと is_done() で誤ってスキップされてしまう。
# ============================================================
PHASE1_EXPERIMENTS_DIR = "experiments_phase1"

PHASE1_RATIOS = {
    # 手法名 -> {データ分布 -> Malicious Ratio一覧}（分布で変えない場合は "default" のみ指定）
    "fedavg": {"default": [0.0, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]},
    "median": {"default": [0.0, 0.1, 0.2, 0.3]},
    "trimmed_mean": {
        "iid": [0.0, 0.1, 0.2, 0.3],
        "non_iid": [0.0, 0.1, 0.2, 0.225, 0.25, 0.275, 0.3],
    },
    # Krumは n>=f+2 であれば計算可能だが、本研究で検討する範囲（0.0-0.3）を0.05刻みにする
    "krum": {"default": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
    # Bulyanは n>=4f+3（n=100 では f<=24 = ratio<=0.24）を満たす範囲のみ
    "bulyan": {"default": [0.0, 0.05, 0.1, 0.15, 0.2]},
}


def phase1_ratios_for(aggregation: str, data_split: str) -> list:
    table = PHASE1_RATIOS[aggregation]
    return table.get(data_split, table.get("default"))


# ============================================================
# sotsuron_plan2.tex 5.2：検知評価スイープ（最小構成）
#
# 更新ベクトルレベルの異常検知を，Non-IID度（Dirichlet alpha）× 攻撃 の2軸で評価する。
# 集約手法は検知に非依存なため fedavg 固定。悪性割合は 0.2 固定。
# 攻撃なし（none）は benign_spread の基準値を取るために各 alpha で1条件だけ回す。
# 出力先は experiments_detection/ に分ける（検知ログ logs/detection_updates.csv を新規生成）。
# ============================================================
DETECTION_EXPERIMENTS_DIR = "experiments_detection"
DETECTION_AGGREGATIONS = ["fedavg"]
DETECTION_ALPHAS = [100.0, 0.5, 0.1]      # 100≒IID / 0.5 中程度 / 0.1 強い不均一
DETECTION_ATTACKS = ["none", "sign_flip", "alie"]
DETECTION_RATIO = 0.2
DETECTION_TRIALS = [1, 2, 3]
DETECTION_NUM_CLIENTS = 20
DETECTION_NUM_ROUNDS = 30
DETECTION_ALIE_Z = 1.0


def detection_conditions():
    """検知評価スイープの条件を dict で列挙する。"""
    for aggregation in DETECTION_AGGREGATIONS:
        for alpha in DETECTION_ALPHAS:
            for attack in DETECTION_ATTACKS:
                ratio = 0.0 if attack == "none" else DETECTION_RATIO
                for trial in DETECTION_TRIALS:
                    yield {
                        "aggregation": aggregation,
                        "alpha": alpha,
                        "attack": attack,
                        "ratio": ratio,
                        "trial": trial,
                    }


def detection_condition_name(p: dict) -> str:
    return (
        f"{p['aggregation']}_dirichlet_a{p['alpha']}_{p['attack']}"
        f"_ratio{p['ratio']}_trial{p['trial']}"
    )


def run_one_detection(p: dict, federation: str) -> str:
    """検知評価スイープの1条件を実行する。戻り値は "done" / "skipped" / "failed"。"""
    name = detection_condition_name(p)
    output_dir = os.path.join(DETECTION_EXPERIMENTS_DIR, name)

    if is_done(output_dir):
        print(f"[SKIP] {name}（{output_dir}/final_model.pt が既に存在＝完了済み）")
        return "skipped"

    os.makedirs(output_dir, exist_ok=True)

    run_config = (
        f'num-server-rounds={DETECTION_NUM_ROUNDS} '
        f'aggregation-method="{p["aggregation"]}" '
        f'data-split="dirichlet" '
        f'dirichlet-alpha={p["alpha"]} '
        f'malicious-ratio={p["ratio"]} '
        f'malicious-seed={p["trial"]} '
        f'trial={p["trial"]} '
        f'attack-type="{p["attack"]}" '
        f'alie-z={DETECTION_ALIE_Z} '
        f'num-clients={DETECTION_NUM_CLIENTS} '
        f'output-dir="{output_dir}"'
    )

    # 新しいflwr（SuperLink接続方式）では --federation-config は非推奨で無視される。
    # クライアント数は main() 側であらかじめ ensure_simulation_config() で設定しておく。
    cmd = [
        *_flwr_cmd_prefix(), "run", ".", federation,
        "--run-config", run_config,
        "--stream",
    ]

    print(f"[RUN] {name}")
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", env=_utf8_env()
    )

    if is_done(output_dir):
        print(f"[DONE] {name}")
        return "done"

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    output_tail = "\n".join(combined_output.strip().splitlines()[-30:])
    print(f"[FAILED] {name}")
    print(output_tail)
    record_failure(name, cmd, output_tail, experiments_dir=DETECTION_EXPERIMENTS_DIR)
    return "failed"


def phase1_conditions():
    """(aggregation, data_split, ratio, trial) を Phase 1 の条件分だけ列挙する。"""
    for aggregation in AGGREGATION_METHODS:
        for data_split in DATA_SPLITS:
            for ratio in phase1_ratios_for(aggregation, data_split):
                for trial in TRIALS:
                    yield aggregation, data_split, ratio, trial


def condition_name(aggregation: str, data_split: str, ratio: float, trial: int) -> str:
    return f"{aggregation}_{data_split}_ratio{ratio}_trial{trial}"


def is_done(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "final_model.pt"))


def record_failure(name: str, cmd: list, output_tail: str, experiments_dir: str = EXPERIMENTS_DIR) -> None:
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


def run_one(
    aggregation: str,
    data_split: str,
    ratio: float,
    trial: int,
    num_clients: int,
    num_rounds: int,
    federation: str,
    experiments_dir: str = EXPERIMENTS_DIR,
) -> str:
    """1条件を実行する。戻り値は "done" / "skipped" / "failed"。"""
    name = condition_name(aggregation, data_split, ratio, trial)
    output_dir = os.path.join(experiments_dir, name)

    if is_done(output_dir):
        print(f"[SKIP] {name}（{output_dir}/final_model.pt が既に存在＝完了済み）")
        return "skipped"

    os.makedirs(output_dir, exist_ok=True)

    # --run-config はTOML文字列として解釈されるため、Windowsのバックスラッシュ区切り
    # パスをそのまま埋め込むと "\f" 等がエスケープシーケンスとして解釈され、
    # パスが壊れてしまう（例: "\fedavg" -> フォームフィード文字+"edavg"）。
    # スラッシュ区切りに変換して渡す（Windows/Pythonのos.path系はスラッシュも受理する）。
    output_dir_for_config = output_dir.replace(os.sep, "/")

    run_config = (
        f'num-server-rounds={num_rounds} '
        f'aggregation-method="{aggregation}" '
        f'data-split="{data_split}" '
        f'malicious-ratio={ratio} '
        f'malicious-seed={trial} '
        f'trial={trial} '
        f'attack-type="{ATTACK_TYPE}" '
        f'num-clients={num_clients} '
        f'output-dir="{output_dir_for_config}"'
    )

    # 新しいflwr（SuperLink接続方式）では --federation-config は非推奨で無視される。
    # クライアント数はあらかじめ ensure_simulation_config() で federation 側に設定しておく。
    cmd = [
        *_flwr_cmd_prefix(), "run", ".", federation,
        "--run-config", run_config,
        "--stream",  # --streamを付けないと flwr run は完了を待たずに即座に返ってしまう
    ]

    print(f"[RUN] {name}")
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", env=_utf8_env()
    )

    if is_done(output_dir):
        print(f"[DONE] {name}")
        return "done"

    # final_model.pt が無ければ失敗とみなす（終了コードは当てにできないため）
    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    output_tail = "\n".join(combined_output.strip().splitlines()[-30:])
    print(f"[FAILED] {name}")
    print(output_tail)
    record_failure(name, cmd, output_tail, experiments_dir=experiments_dir)
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="実験計画書7.6：全実験条件の自動実行")
    parser.add_argument("--quick", action="store_true", help="動作確認用の小規模設定で試す")
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="report/mechanism_analysis_plan.tex のPhase 1（全条件＋臨界点探索用のRatio細分化、trial数はTRIALS）を実行する。"
        "出力先は experiments/ とは別の experiments_phase1/ に分ける。",
    )
    parser.add_argument(
        "--detection",
        action="store_true",
        help="sotsuron_plan2.tex 5.2 の検知評価スイープ（Dirichlet alpha × 攻撃、fedavg固定、"
        f"{DETECTION_NUM_CLIENTS}クライアント・{DETECTION_NUM_ROUNDS}ラウンド・trial{DETECTION_TRIALS}）を実行する。"
        "出力先は experiments_detection/。",
    )
    parser.add_argument("--federation", default="local", help="flwr run に渡すSuperLink接続名（~/.flwr/config.tomlに定義されたもの）")
    parser.add_argument(
        "--aggregations",
        default=None,
        help="実行対象のaggregation_methodをカンマ区切りで指定（未指定時は全手法）。"
        "例：--aggregations fedavg で再実験など一部手法のみ再実行したい場合に使う。",
    )
    args = parser.parse_args()

    selected_aggregations = None
    if args.aggregations is not None:
        selected_aggregations = {a.strip() for a in args.aggregations.split(",")}

    # --- sotsuron_plan2.tex 5.2：検知評価スイープ（独立した実行パス） ---
    if args.detection:
        conditions = list(detection_conditions())
        if selected_aggregations is not None:
            conditions = [c for c in conditions if c["aggregation"] in selected_aggregations]
        total = len(conditions)
        print(f"=== 検知評価スイープ 条件数: {total}（出力先: {DETECTION_EXPERIMENTS_DIR}/） ===\n")

        ensure_simulation_config(args.federation, DETECTION_NUM_CLIENTS)

        counts = {"done": 0, "skipped": 0, "failed": 0}
        for i, p in enumerate(conditions, start=1):
            print(f"[{i}/{total}]", end=" ")
            counts[run_one_detection(p, args.federation)] += 1
        print("\n=== 検知評価スイープ終了 ===")
        print(f"完了: {counts['done']} / スキップ: {counts['skipped']} / 失敗: {counts['failed']} / 合計: {total}")
        if counts["failed"] > 0:
            print(f"失敗した条件は {os.path.join(DETECTION_EXPERIMENTS_DIR, FAILURES_CSV_NAME)} を参照してください。")
        return

    if args.quick:
        num_clients = 10
        num_rounds = 1
        experiments_dir = EXPERIMENTS_DIR
        conditions = [
            (aggregation, data_split, ratio, trial)
            for aggregation in ["fedavg", "median"]
            for data_split in ["iid"]
            for ratio in [0.0, 0.2]
            for trial in [1]
        ]
    elif args.phase1:
        num_clients = NUM_CLIENTS
        num_rounds = NUM_SERVER_ROUNDS
        experiments_dir = PHASE1_EXPERIMENTS_DIR
        conditions = list(phase1_conditions())
    else:
        num_clients = NUM_CLIENTS
        num_rounds = NUM_SERVER_ROUNDS
        experiments_dir = EXPERIMENTS_DIR
        conditions = [
            (aggregation, data_split, ratio, trial)
            for aggregation in AGGREGATION_METHODS
            for data_split in DATA_SPLITS
            for ratio in MALICIOUS_RATIOS
            for trial in TRIALS
        ]

    if selected_aggregations is not None:
        conditions = [c for c in conditions if c[0] in selected_aggregations]

    total = len(conditions)
    print(f"=== 実験条件数: {total}（出力先: {experiments_dir}/） ===\n")

    ensure_simulation_config(args.federation, num_clients)

    counts = {"done": 0, "skipped": 0, "failed": 0}
    for i, (aggregation, data_split, ratio, trial) in enumerate(conditions, start=1):
        print(f"[{i}/{total}]", end=" ")
        status = run_one(
            aggregation, data_split, ratio, trial,
            num_clients, num_rounds, args.federation,
            experiments_dir=experiments_dir,
        )
        counts[status] += 1

    print("\n=== 実験終了 ===")
    print(f"完了: {counts['done']} / スキップ（完了済み）: {counts['skipped']} / "
          f"失敗: {counts['failed']} / 合計: {total}")
    if counts["failed"] > 0:
        print(f"失敗した条件は {os.path.join(experiments_dir, FAILURES_CSV_NAME)} を参照してください。")


if __name__ == "__main__":
    main()
