"""
log_output.py

実験ログの出力（サーバ側で使用）。

出力するファイル（すべて LOG_DIR の下。スイープ時は <output-root>/_logs/）:
  run_config.txt          : 実行ごとの設定の全項目（1実行1ブロック）
  results.csv             : ラウンドごとの Accuracy / Loss（1実行×1ラウンドで1行）
  summary.csv             : 最終ラウンドの Accuracy / Loss（1実行で1行）
  detection_updates.csv   : 検知器スコアと正解ラベル（1実行×1ラウンド×1クライアントで1行）
  detection_metrics.csv   : 検知性能指標（1実行×1ラウンド×1検知器で1行）。analyze_detection.py の入力

どの CSV にも実験条件（alpha・攻撃・z・trial など）の列を持たせ，
全条件が1つのファイルに追記されても条件ごとに切り分けられるようにしている。
"""

# 【このファイルの読み方】
#   実験の記録（ログ）をファイルに書き出す関数をまとめたファイル。計算は何もしていない。
#   渡された値を，決まった列の順番で，ファイルの末尾に1行ずつ書き足していくだけである。
#   ここで書き出したファイルが，実験の「結果」のすべてになる（分析はこのファイルたちを読んで行う）。
#
#       関数                        書き出すファイル          呼び出す側                 いつ
#       configure_log_dir           （書き出し先を決める）    server_app.py              実行の最初に1回
#       log_run_config              run_config.txt            server_app.py              実行の最初に1回
#       log_detection_csv           detection_updates.csv     robust_strategy.py         毎ラウンド（クライアントの台数ぶんの行）
#       log_detection_metrics_csv   detection_metrics.csv     robust_strategy.py         毎ラウンド（検知器の数＝4行）
#       log_results_csv             results.csv               server_app.py              毎ラウンド（1行）
#       log_summary_csv             summary.csv               server_app.py              実行の最後に1回（1行）
#       print_round_summary         （画面に表示するだけ）    server_app.py / client_app.py
#
#   「追記」であること
#     どのファイルも，上書きではなく末尾への書き足し（追記）になっている。
#     スイープの54条件は，すべて同じファイルに順番に書き足されていく。
#     どの行がどの条件のものかは，各行に入っている条件の列（alpha，attack_type など）と run_id で見分ける。
#
#   CSV とは
#     値をカンマで区切って並べたテキストファイル。1行目が列の名前（見出し），2行目以降がデータ。
#     表計算ソフトや pandas でそのまま読める。
#
# 【import の説明】
import os                      # ファイルやフォルダを扱う道具（Python 標準）
import csv                     # CSV ファイルを読み書きする道具（Python 標準）
from datetime import datetime  # 日付と時刻を扱う道具（Python 標準）

# 【書き出し先の変数の説明】
#   ログを書き出す場所を覚えておく変数。ファイルの中のどの関数からも参照できる（モジュール変数）。
#   最初は "logs" フォルダになっているが，実行の最初に configure_log_dir() が呼ばれると，
#   スイープのフォルダ（例: experiments_detection_v2/_logs）に切り替わる。
#     os.path.join(A, B) : フォルダ名とファイル名を，区切り文字（/）でつないで1つのパスにする。
#ログの保存先
LOG_DIR = "logs"
CONFIG_TXT = os.path.join(LOG_DIR, "run_config.txt")
RESULTS_CSV = os.path.join(LOG_DIR, "results.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary.csv")


# 【configure_log_dir の説明】
#   書き出し先のフォルダを切り替える。上の変数たちを，新しいフォルダを指すように書き換える。
#     global 名前, ... : 関数の中から，関数の外にある変数を書き換えるための宣言。
#                        これが無いと，関数の中だけで使う別の変数ができてしまい，外の変数は変わらない。
#   検知ログ用の2つの変数（DETECTION_CSV，DETECTION_METRICS_CSV）は，このファイルの下のほうで
#   定義されているが，ここで一緒に書き換えている。
def configure_log_dir(log_dir: str) -> None:
    """
    ログの出力先を切り替える。

    デフォルトの LOG_DIR="logs" はプロジェクト直下の固定パスで、
    run_experiments.py が条件ごとに分ける output_dir とは独立している。
    そのため output_dir を変えても全条件のログが同じ logs/ に混ざってしまい、
    後から解析する際に条件の切り分けや掃除が困難になる（実際に発生した問題）。
    server_app.py から output_dir に対応する場所を渡すことで、
    そのスイープ専用のログディレクトリに自己完結させる。
    """
    global LOG_DIR, CONFIG_TXT, RESULTS_CSV, SUMMARY_CSV
    global DETECTION_CSV, DETECTION_METRICS_CSV

    LOG_DIR = log_dir
    CONFIG_TXT = os.path.join(LOG_DIR, "run_config.txt")
    RESULTS_CSV = os.path.join(LOG_DIR, "results.csv")
    SUMMARY_CSV = os.path.join(LOG_DIR, "summary.csv")
    DETECTION_CSV = os.path.join(LOG_DIR, "detection_updates.csv")
    DETECTION_METRICS_CSV = os.path.join(LOG_DIR, "detection_metrics.csv")


# =========================
# 内部ユーティリティ
# =========================
# 【_ensure_dir の説明】
#   書き出し先のフォルダが無ければ作る。どの書き出しの関数も，最初にこれを呼んでいる。
#   exist_ok=True は「すでにあってもエラーにしない」。
def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# 実験設定（ハイパーパラメータ）記録
# =========================
# 【log_run_config の説明】
#   その実行で使われた設定を，全項目そのまま run_config.txt に書き残す。
#   あとで「この結果は，どの設定で回したものだったか」を確かめるための記録。
#   CSV ではなく，人が読むためのテキスト形式（1実行ぶんを，線で区切った1かたまりにしている）。
#
#   with open(CONFIG_TXT, "a", encoding="utf-8") as f:
#       ファイルを開いて f という名前で扱う。with を使うと，字下げした部分が終わったときに自動で閉じられる。
#       "a" は追記モード（append）。ファイルが無ければ新しく作り，あれば末尾に書き足す。
#       encoding="utf-8" は文字コードの指定（日本語を含んでも文字化けしないように）。
#   f.write("...\n")          : 文字列を書き込む。\n は改行。
#   datetime.now().isoformat(timespec='seconds') : 現在の日時を 2026-10-09T18:03:01 の形の文字列にする。
#   for key in sorted(config): : 設定の名前を，アルファベット順に並べて1つずつ取り出す。
#   f"{key} : {config[key]}"   : 「名前 : 値」の形の1行を作る。
def log_run_config(run_id: str, config: dict):
    """
    1回の実行で実際に使われた設定を，全項目そのまま run_config.txt に追記する。

    config: 設定名 -> 値 の辞書（pyproject.toml の値に --run-config の上書きを反映したもの，
            および悪性クライアントの番号など実行時に決まった値）。
    あとから「どの設定で回した結果か」を run_id で突き合わせて確かめるための記録。
    """
    _ensure_dir()
    with open(CONFIG_TXT, "a", encoding="utf-8") as f:
        f.write("=================================\n")
        f.write(f"run_id : {run_id}\n")
        f.write(f"timestamp : {datetime.now().isoformat(timespec='seconds')}\n")
        for key in sorted(config):
            f.write(f"{key} : {config[key]}\n")
        f.write("=================================\n\n")


# =========================
# 実験結果ログ（実験計画書 7.5 対応）
# =========================
# 【列の定義の説明】
#   CSV の列の名前と順番を，リストで決めている。書き出すときは，必ずこの順番で値を並べる。
#     _CONDITION_COLUMNS : 実験条件を表す7つの列。results.csv と summary.csv の両方で使う。
#     リスト + リスト    : 2つのリストをつなげて，1つのリストにする。
#                          「そのファイル固有の列」のあとに「条件の列」が続く形になる。
#   条件の列を全部のログに入れているのは，全条件が1つのファイルに追記されるため。
#   これが無いと，どの行がどの条件のものか，あとから区別できない。
#
# 【log_results_csv / log_summary_csv の説明】
#   精度（accuracy）と損失（loss）を記録する。サーバがテストデータ 10,000 枚で採点した結果。
#     results.csv : 毎ラウンド1行。学習が進むにつれて精度がどう変わったかを追える。
#     summary.csv : 1回の実行につき1行。最終ラウンドの結果だけ。
#   2つの関数は，書き出すファイルと列が違うだけで，やっていることは同じ。
#
#   file_exists = os.path.exists(RESULTS_CSV) : そのファイルがすでにあるかどうか（True / False）。
#   with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
#       追記モードで開く。newline="" は，CSV で余計な空行が入らないようにするための決まり文句。
#   w = csv.writer(f)          : 「リストを渡すと，カンマ区切りの1行にして書いてくれる係」を作る。
#   if not file_exists: w.writerow(_RESULTS_COLUMNS)
#       : ファイルがまだ無かった（＝これが最初の書き込み）ときだけ，1行目に列の名前を書く。
#         2回目以降は見出しを書かないので，見出しはファイルの先頭に1回だけ入る。
#   [condition[c] for c in _CONDITION_COLUMNS]
#       : 条件の辞書から，列の順番どおりに値を取り出して並べる（リスト内包表記）。
#         condition["名前"] の形なので，辞書に項目が欠けていればエラーになる（黙って空欄にはならない）。
#   値が None のもの（ALIE 以外の alie_z など）は，CSV では空欄になる。
_CONDITION_COLUMNS = [
    "aggregation_method",
    "data_split",
    "alpha",
    "attack_type",
    "alie_z",            # ALIE の z（attack_type="alie" 以外は空）
    "malicious_ratio",
    "trial",
]

_RESULTS_COLUMNS = ["run_id", "round", "accuracy", "loss_func"] + _CONDITION_COLUMNS

_SUMMARY_COLUMNS = ["run_id", "final_round", "final_accuracy", "final_loss"] + _CONDITION_COLUMNS


def log_results_csv(run_id: str, round_: int, accuracy: float, loss_func: float, condition: dict):
    """
    ラウンドごとの Accuracy/Loss を、実験条件つきで logs/results.csv に追記する。

    condition: _CONDITION_COLUMNS のキーを持つ辞書（実験条件）。
    """
    _ensure_dir()
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_RESULTS_COLUMNS)
        w.writerow([run_id, round_, accuracy, loss_func] + [condition[c] for c in _CONDITION_COLUMNS])


def log_summary_csv(run_id: str, final_round: int, final_accuracy: float, final_loss: float, condition: dict):
    """
    1回の実行につき1行、最終ラウンドの Accuracy/Loss を logs/summary.csv に追記する。

    condition: _CONDITION_COLUMNS のキーを持つ辞書（実験条件）。
    """
    _ensure_dir()
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_SUMMARY_COLUMNS)
        w.writerow(
            [run_id, final_round, final_accuracy, final_loss] + [condition[c] for c in _CONDITION_COLUMNS]
        )


# =========================
# 検知評価用ログ（sotsuron_plan2.tex 4.2--4.5 対応）
# =========================
# 【この部分の説明】
#   この研究の中心になる記録。2つのファイルがある。
#     detection_updates.csv : 生の記録。1クライアント × 1ラウンドで1行（20台 × 50ラウンド × 54条件 = 54,000 行）。
#                             各クライアントの更新の大きさ，4つの検知器のスコア，悪性かどうか。
#     detection_metrics.csv : 成績の記録。1検知器 × 1ラウンドで1行（4 × 50 × 54 = 10,800 行）。
#                             ROC-AUC などの検知性能指標。analyze_detection.py はこちらだけを読む。
#   成績は生の記録から計算できるので，生の記録があれば，あとから別の指標で計算し直すこともできる。
#
#   列の一覧（_DETECTION_COLUMNS / _DETECTION_METRICS_COLUMNS）の読み方
#     実験条件      : run_id，round，aggregation_method，data_split，alpha，attack_type，alie_z，malicious_ratio，trial
#     クライアント  : client_pos（表の何行目か），partition_id（クライアント番号），is_malicious（正解。悪性なら 1）
#     更新と検知    : delta_norm（更新ベクトルの長さ），d1_norm 〜 d4_spectral（検知器のスコア）
#     そのラウンドの指標 : benign_spread，push_ratio，分布構造の診断指標6つ（同じラウンドの行では全部同じ値）
#     成績          : detector（どの検知器か），auc，ap，recall_at_f，precision_at_f
#     参考の値      : mean_dist_unfiltered，mean_dist_filtered（記録するだけで，卒業論文では扱わない）
#
# 【log_detection_csv / log_detection_metrics_csv の説明】
#   rows は辞書のリスト。辞書1つが CSV の1行になる（robust_strategy.py が作って渡してくる）。
#     if not rows: return  : リストが空なら，何もせずに終わる。
#     for r in rows:       : 辞書を1つずつ取り出して，1行ずつ書く。
#     [r[col] for col in _DETECTION_COLUMNS]
#         : 列の順番どおりに，辞書から値を取り出して並べる。
#           r[col] は，辞書にその名前が無いとエラーになる取り出し方。項目が欠けた行を，
#           空欄のまま黙って書いてしまわないようにしている（以前は r.get(col) で，欠けていても空欄で書いていた）。
#
#   【運用の決まり】コードを変えてログの列が変わったら，出力先（pyproject.toml の output-root）の名前を変える。
#     見出し（列の名前）はファイルを最初に作るときに1回だけ書かれ，2回目以降は中身を確かめずに書き足す。
#     そのため，列を変えた後に同じフォルダへ追記すると，見出しと合わない行が混ざってしまう。
DETECTION_CSV = os.path.join(LOG_DIR, "detection_updates.csv")

_DETECTION_COLUMNS = [
    "run_id",
    "round",
    "aggregation_method",
    "data_split",
    "alpha",
    "attack_type",
    "malicious_ratio",
    "trial",
    "client_pos",       # updates 配列内での位置（0..n-1）
    "partition_id",      # 実際のクライアントID
    "is_malicious",      # 正解ラベル（評価専用。検知器には渡さない）
    "delta_norm",        # ||g_i||（更新ベクトルのノルム）
    "d1_norm",           # 検知器スコア（大きいほど悪性らしい）
    "d2_cosine",
    "d3_kmeans",
    "d4_spectral",
    "benign_spread",     # そのラウンドの正常クラスタの広がり（全行で同じ値）
    "push_ratio",        # そのラウンドの push_ratio（全行で同じ値。該当なしは空）
    "alie_z",            # ALIE の z（attack_type="alie" 以外は空）
    # --- 分布構造の診断指標（detection.STRUCTURE_KEYS。全行で同じ値） ---
    "pc1_ratio",
    "mal_cos_pc1",
    "pc1_ratio_all",
    "mal_cos_pc1_all",
    "mal_rank_mean",
    "mal_rank_std",
]


def log_detection_csv(rows: list):
    """
    集約直前（攻撃適用後）の各クライアント更新について，検知器スコアと正解ラベルを
    logs/detection_updates.csv へ1クライアント1行で追記する。

    rows: dict のリスト。各 dict は _DETECTION_COLUMNS のキーを持つ。
    """
    if not rows:
        return
    _ensure_dir()
    file_exists = os.path.exists(DETECTION_CSV)
    with open(DETECTION_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_DETECTION_COLUMNS)
        for r in rows:
            w.writerow([r[col] for col in _DETECTION_COLUMNS])


# 検知性能指標のログ（1ラウンド×1検知器で1行）。
# analyze_detection.py はこのファイルだけを読む（指標の計算式は detection.py の1か所に置き，
# 解析側には持たせない）。あとから別の指標を計算し直せるよう，1クライアント1行の
# detection_updates.csv も引き続き残す。
DETECTION_METRICS_CSV = os.path.join(LOG_DIR, "detection_metrics.csv")

_DETECTION_METRICS_COLUMNS = [
    "run_id",
    "round",
    "aggregation_method",
    "data_split",
    "alpha",
    "attack_type",
    "alie_z",            # ALIE の z（attack_type="alie" 以外は空）
    "malicious_ratio",
    "trial",
    "n_clients",
    "n_malicious",       # f。recall_at_f / precision_at_f の k に使う
    "benign_spread",
    "push_ratio",
    # --- 分布構造の診断指標（detection.STRUCTURE_KEYS） ---
    "pc1_ratio",
    "mal_cos_pc1",
    "pc1_ratio_all",
    "mal_cos_pc1_all",
    "mal_rank_mean",
    "mal_rank_std",
    "detector",          # d1_norm / d2_cosine / d3_kmeans / d4_spectral
    "auc",               # ROC-AUC（悪性0人のラウンドは nan）
    "ap",                # average precision
    "recall_at_f",       # スコア上位 f 件を悪性と判定したときの recall
    "precision_at_f",    # 同 precision
    # --- 参考の値（記録するだけ。卒業論文では扱わない。detection.filtered_mean_distance のメモを参照）---
    "mean_dist_unfiltered",  # 何も取り除かない場合の，全クライアントの平均と良性の重心との距離（同じラウンドでは同じ値）
    "mean_dist_filtered",    # その検知器のスコア上位 f 件を取り除いた場合の，残りの平均と良性の重心との距離
]


def log_detection_metrics_csv(rows: list):
    """
    そのラウンドの検知性能指標を logs/detection_metrics.csv へ1検知器1行で追記する。

    rows: dict のリスト。各 dict は _DETECTION_METRICS_COLUMNS のキーを持つ。
    """
    if not rows:
        return
    _ensure_dir()
    file_exists = os.path.exists(DETECTION_METRICS_CSV)
    with open(DETECTION_METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(_DETECTION_METRICS_COLUMNS)
        for r in rows:
            w.writerow([r[col] for col in _DETECTION_METRICS_COLUMNS])


# =========================
# ターミナル用 簡易表示
# =========================
# 【print_round_summary の説明】
#   ラウンドの結果を，画面に1行で表示する。ファイルには書かない。実行中の様子を見るためのもの。
#     表示の例: [ROUND 3] | accuracy=0.9113 | loss_func=0.2874
#   サーバからは accuracy と loss_func，クライアントからは train_loss や eval_loss と eval_acc が渡される。
#   渡されたものだけを表示する。
#
#   train_loss: float | None = None
#       型ヒントは「小数か None」，既定値は None。渡されなかった項目は None になる。
#       これは設定値ではなく，「その項目は表示しない」ことを表すための既定値。
#   parts = [f"[ROUND {round_}]"]  : 表示する部品を入れていくリスト。最初はラウンド番号だけ。
#       round_ と末尾に _ が付いているのは，Python にもともとある round という関数と名前がぶつからないようにするため。
#   if train_loss is not None:     : その項目が渡されていれば，
#       parts.append(f"train_loss={train_loss:.4f}") : 「名前=値」の文字列をリストに足す。:.4f は小数第4位まで。
#   " | ".join(parts)              : リストの部品を " | " でつないで，1つの文字列にする。
def print_round_summary(
    round_: int,
    train_loss: float | None = None,
    eval_loss: float | None = None,
    eval_acc: float | None = None,
    accuracy: float | None = None,
    loss_func: float | None = None,
):
    parts = [f"[ROUND {round_}]"]
    if train_loss is not None:
        parts.append(f"train_loss={train_loss:.4f}")
    if eval_loss is not None:
        parts.append(f"eval_loss={eval_loss:.4f}")
    if eval_acc is not None:
        parts.append(f"eval_acc={eval_acc:.4f}")
    if accuracy is not None:
        parts.append(f"accuracy={accuracy:.4f}")
    if loss_func is not None:
        parts.append(f"loss_func={loss_func:.4f}")
    print(" | ".join(parts))
