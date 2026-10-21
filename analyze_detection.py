"""
analyze_detection.py

sotsuron_plan2.tex 4.5 / 5 対応：検知評価スイープの後処理（集計と作図）。

【このファイル1本で完結する】
実験コード（Flower / fl_detection / detection.py / pyproject.toml）には依存しない。
必要なのは numpy / pandas / matplotlib と，実験が出力したログのフォルダだけ。
ROC-AUC などの検知性能指標は実験中に計算済み（robust_strategy.py が detection.py を
使って detection_metrics.csv に記録する）なので，ここでは読み込んで集計・作図するだけ。

入力 : <logdir>/detection_metrics.csv
       <logdir>/results.csv（あれば。ラウンドごとのテスト精度。無ければ精度の図と表は作らない）
       <logdir>/detection_updates.csv（あれば。1クライアント1行のスコア。無ければ符号反転の分離の図と表は作らない）
       1行 =（実行, ラウンド, 検知器）。ROC-AUC / AP / recall@f / precision@f と，
       spread / push_ratio / 分布構造の診断指標を同じ行に持つ。
出力 : <outdir>/（既定は <logdir>/analysis/）
  - detection_auc_by_round.png
      検知器ごとに「AUC 対 ラウンド」を (alpha, 攻撃) 別に描く（trial 平均±標準偏差）
  - detection_pushratio_vs_auc.png
      符号反転だけを対象にした，横軸 push_ratio・縦軸 AUC の散布図（中心仮説の検証用）
  - detection_auc_by_round_<検知器>.png / detection_pushratio_vs_auc_<検知器>.png
      上の2つの図を，検知器ごとに1枚ずつに分けたもの（d1_norm / d2_cosine / d3_kmeans / d4_spectral）
  - detection_pushratio_corr_table.csv
      符号反転での push_ratio と AUC の順位相関（全体と alpha ごと）
  - detection_auc_by_z.png
      ALIE だけを対象にした，横軸 z・縦軸 AUC の図（最終ラウンド付近の平均，trial 平均±標準偏差）
  - detection_final_auc_table.csv
      最終ラウンド付近の AUC 一覧（alpha × 攻撃 × 検知器。trial 間の平均と標準偏差）
  - detection_structure_table.csv
      最終ラウンド付近の spread / push_ratio / 構造指標と AUC の対応表
  - detection_segment_auc_table.csv
      ラウンドの区間（第1-5, 6-15, 16-30, 31-50ラウンド）ごとの AUC（trial 平均。検知器 × 攻撃 × alpha）
  - detection_segment_pushratio_table.csv
      同じ区間ごとの push_ratio（trial 平均。攻撃 × alpha）
  - signflip_d2_score_by_round.png
      符号反転での D2 スコア（1 - cos）のラウンド推移を，正常更新と悪性更新に分けて描く（trial 平均±標準偏差）
  - signflip_separation_table.csv
      符号反転で悪性と正常が分かれる様子の表（D2 スコアの平均，ラウンド内で完全に分かれた割合，
      悪性方向と全更新の第1主成分の |cos|。第1-5ラウンドと最終ラウンド付近）
  - signflip_norm_ratio_by_round.png
      符号反転での悪性/正常のノルムの比のラウンド推移と，攻撃なしで同じ4台を悪性とみなした場合の比（対照）
  - signflip_norm_dispersion_table.csv
      符号反転の最終ラウンド付近での，ノルムの比，正常更新のノルムの変動係数（攻撃なしとの比較），
      D1・D4 が順位を誤った組の割合とその内訳（中央値より小さい／大きい正常更新によるもの），
      正常更新どうしでの D4 のスコアとノルムの順位相関
  - signflip_norm_cv_by_segment_table.csv
      正常更新のノルムの変動係数を，ラウンドの区間ごとに，符号反転と攻撃なしで比べた表
  - alie_d2_score_by_z.png / alie_d2_score_by_z_table.csv
      ALIE の z ごとの，最終ラウンド付近の D2 スコア（正常と悪性。trial 平均±標準偏差）。表の最後の行は攻撃なしの正常
  - alie_norm_ratio_by_z_table.csv
      ALIE の z ごとの，最終ラウンド付近の悪性/正常のノルムの比（D1 の U 字型の説明用）
  - alie_shell_by_z.png / alie_shell_table.csv
      卒論 9.3（高次元の殻）。z ごとの距離の比（push_ratio）・ノルムの比と，正常の殻の厚み（ノルムと距離の変動係数）
  - accuracy_by_round.png
      グローバルモデルのテスト精度のラウンド推移（alpha ごと。trial 平均±標準偏差）
  - accuracy_table.csv
      最終ラウンドの精度と，最後の数ラウンドでの精度の上がり幅（学習が収束しているかの確認）

ALIE は z ごとに別の攻撃条件（alie_z0.5 など）として扱う。

【push_ratio の分析から ALIE を除いている理由】
ALIE の悪性更新は「良性の平均 - z × 良性の標準偏差」で作るので，push_ratio は作り方の時点で
ほぼ z に等しくなる（sotsuron_plan2.tex の式 push_ratio = z * RMS / Mean >= z）。
つまり ALIE では push_ratio は測って分かる量ではなく，z で決めた量であり，alpha を変えても動かない。
そこで，push_ratio と AUC の関係は符号反転だけで調べ，ALIE は z を横軸にして別の図で見る。

使い方:
  python3 analyze_detection.py <logdir>
  python3 analyze_detection.py <logdir> --outdir <dir> --last-k 5

  例: python3 analyze_detection.py experiments_detection_v2/_logs
"""

# 【このファイルの読み方】
#   実験が終わったあとに，結果を集計して図と表にする補助スクリプト。
#   研究の中身（攻撃・検知・学習）は入っておらず，新しい計算もほとんどしていない。
#   ROC-AUC などの成績は，実験中に計算されて detection_metrics.csv に入っている。
#   ここでやっているのは，それを読んで「平均を取る」「並べ替える」「図に描く」ことだけである。
#
#   全体の流れ（一番下の main() が，上から順に実行する）
#       1. ログのフォルダから detection_metrics.csv を読む        load_metrics
#       2. 図1：AUC がラウンドとともにどう変わるか               plot_auc_by_round（検知器ごとの版は plot_auc_by_round_by_detector）
#       3. 図2：push_ratio と AUC の関係（符号反転だけ）         plot_pushratio_vs_auc（検知器ごとの版は plot_pushratio_vs_auc_by_detector）
#       4. 表1：push_ratio と AUC の順位相関（符号反転だけ）     pushratio_corr_table
#       5. 図3：z と AUC の関係（ALIE だけ）                     plot_auc_by_z
#       6. 表2：最後の数ラウンドの平均 AUC                       final_auc_table
#       7. 表3：分布構造の診断指標と AUC の対応                  structure_table
#       8. 図4：テスト精度のラウンド推移                         plot_accuracy_by_round（results.csv があるときだけ）
#       9. 表4：最終精度と，最後の数ラウンドでの上がり幅         accuracy_table（同上）
#      10. 図5：符号反転での D2 スコアのラウンド推移（正常と悪性） plot_signflip_d2_score_by_round（detection_updates.csv があるときだけ）
#      11. 表5：符号反転で悪性と正常が分かれる様子               signflip_separation_table（同上）
#      12. 図6：符号反転でのノルムの比のラウンド推移（攻撃なしと比較） plot_signflip_norm_ratio_by_round（同上）
#      13. 表6：符号反転でのノルムのばらつきと D1・D4 の誤り          signflip_norm_dispersion_table（同上）
#      14. 表7：正常更新のノルムの変動係数（ラウンドの区間ごと）  signflip_norm_cv_by_segment_table（同上）
#      15. 図7・表8：ALIE の z ごとの D2 スコア（正常と悪性）     plot_alie_d2_score_by_z / alie_d2_score_by_z_table（同上）
#      16. 表9：ALIE の z ごとのノルムの比                        alie_norm_ratio_by_z_table（同上）
#      17. 図8・表10：ALIE の殻（距離の比・ノルムの比・殻の厚み）  plot_alie_shell_by_z / alie_shell_table（同上）
#
#   使い方
#       python analyze_detection.py <detection_metrics.csv が入っているフォルダ>
#       例: python analyze_detection.py experiments_detection_v2/_logs
#   図と表は，指定したフォルダの中の analysis/ に出力される。
#
#   読み込むデータの形
#       表（pandas の DataFrame）。1行が「ある実行の，あるラウンドの，ある検知器の成績」。
#       300条件 × 50ラウンド × 検知器4つ = 60,000 行。
#       列は，実験条件（alpha，attack_type，trial など），round，detector，auc，push_ratio など。
#
# 【このファイルで何度も出てくる pandas の書き方】
#   df["列名"]                     : 表から1列を取り出す。
#   df[df["列名"] == 値]           : その列が値に等しい行だけを取り出す（条件で絞り込む）。
#   df[(条件A) & (条件B)]          : 両方の条件を満たす行だけを取り出す（& は「かつ」）。
#   df.groupby("列名")["auc"].mean() : その列の値ごとにグループに分け，グループごとに auc の平均を取る。
#   .unique()                      : 列に現れる値の種類（重複を除いたもの）。
#   .reset_index()                 : グループ分けの結果を，普通の表の形に戻す。
#
# 【import の説明】
import argparse                 # コマンドの後ろに付ける指定を読み取る（Python 標準）
import os                       # ファイルやフォルダを扱う（Python 標準）

import numpy as np              # 数値計算のライブラリ
import pandas as pd             # 表の形のデータを扱うライブラリ（CSV の読み込みと集計に使う）
import matplotlib               # 図を描くライブラリ

# 図を画面に表示せず，ファイルに保存するだけにする設定（"Agg"）。
# 画面のない環境でも動くようにするためで，pyplot を読み込む前に指定する決まり。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 【定数の説明】（ファイル全体で使う，変わらない値。名前を大文字で書く慣習）
#   METRICS_CSV_NAME : 読み込むファイルの名前。
#   RESULTS_CSV_NAME : ラウンドごとのテスト精度のファイルの名前（無くても動く）。
#   UPDATES_CSV_NAME : 1クライアント1行の検知スコアのファイルの名前（無くても動く）。
#   DETECTORS        : 検知器の名前（ログの detector 列に入っている値）。図や表の並び順にもなる。
#   DETECTOR_LABELS  : 図に表示するときの，検知器の見やすい名前。
#   CONDITION_KEYS   : 実験条件を表す列の名前。この7つがすべて同じ行は，同じ条件の実行とみなす。
#   STRUCTURE_COLS   : 分布構造の診断指標の列の名前。
#   PAPER_RC         : 論文に載せる図の見た目の設定（字の大きさや目盛りの向きなど）をまとめた辞書。
#                      with plt.rc_context(PAPER_RC): の中で描いた図にだけ適用される。
METRICS_CSV_NAME = "detection_metrics.csv"
RESULTS_CSV_NAME = "results.csv"
UPDATES_CSV_NAME = "detection_updates.csv"
DETECTORS = ["d1_norm", "d2_cosine", "d3_kmeans", "d4_spectral"]
DETECTOR_LABELS = {
    "d1_norm": "D1 norm",
    "d2_cosine": "D2 cosine",
    "d3_kmeans": "D3 k-means",
    "d4_spectral": "D4 spectral",
}
CONDITION_KEYS = [
    "aggregation_method",
    "data_split",
    "alpha",
    "attack_type",
    "alie_z",
    "malicious_ratio",
    "trial",
]
# 分布構造の診断指標の列名（ログの列名。実験側では detection.STRUCTURE_KEYS）。
# ログに無い列は空欄として扱うので，食い違ってもエラーにはならずその列が空になる。
STRUCTURE_COLS = [
    "pc1_ratio",
    "mal_cos_pc1",
    "pc1_ratio_all",
    "mal_cos_pc1_all",
    "mal_rank_mean",
    "mal_rank_std",
]


PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


# 【alpha_styles の説明】
#   alpha ごとの線の見た目（色，線の種類，印の形）を決めて，辞書で返す。
#   どの図でも同じ alpha が同じ見た目になるように，ここ1か所で決めている。
#   alpha は大きい順（1000 → 5 → 1）に，赤の実線・青の破線・緑の一点鎖線を割り当てる。
#   色だけでなく線の種類と印でも区別するのは，白黒で印刷しても読めるようにするため。
#     sorted(alphas, reverse=True) : 大きい順に並べる。
#     enumerate(...)               : 「何番目か」と「値」を一緒に取り出す。
#     i % 3                        : i を 3 で割った余り。alpha が4つ以上あっても，見た目を使い回せる。
def alpha_styles(alphas) -> dict:
    """alpha -> (色, 線種, マーカー) の辞書を返す（alpha の大きい順に割り当てる）。"""
    colors = ["tab:red", "tab:blue", "tab:green"]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    return {
        a: (colors[i % 3], linestyles[i % 3], markers[i % 3])
        for i, a in enumerate(sorted(alphas, reverse=True))
    }


# 【attack_label の説明】
#   図や表で使う，攻撃条件の名前を作る。ALIE は z の値ごとに別の条件として扱いたいので，
#   名前の後ろに z を付ける（例: alie_z0.5）。それ以外の攻撃は，名前をそのまま返す。
#     pd.notna(alie_z)    : 値が空（NaN）でなければ True。
#     f"alie_z{alie_z:g}" : :g は，余計な 0 を付けずに数値を表示する書き方（0.50 ではなく 0.5）。
def attack_label(attack_type: str, alie_z) -> str:
    """図表で攻撃条件を区別するラベル（ALIE は z ごとに別条件として扱う）。"""
    if attack_type == "alie" and pd.notna(alie_z):
        return f"alie_z{alie_z:g}"
    return attack_type


# ============================================================
# 1) 実験が出力した指標ログを読む
# ============================================================
# 【keep_last_runs の説明】
#   ログは追記されていくので，ある条件が途中で失敗して回し直された場合，
#   失敗した実行の行と，回し直した実行の行の両方がファイルに残っている。
#   そのまま平均を取ると，同じ条件が二重に数えられる。そこで，条件ごとに
#   「最後に書かれた実行」だけを残し，それより前の実行の行は捨てる。
#   detection_metrics.csv と results.csv の両方で同じ処理をするので，関数にまとめている。
#     df.groupby(CONDITION_KEYS, dropna=False)["run_id"].transform("last")
#         : 条件が同じ行をグループにし，各行に「そのグループで最後に出てくる run_id」を対応させる。
#           dropna=False は，alie_z が空の条件（ALIE 以外）もグループとして扱う指定。
#     stale = df["run_id"] != last_run : 最後の実行ではない行に True が付く。
#     df[~stale]                       : ~ は True と False を入れ替える。つまり，最後の実行の行だけを残す。
#     who : 画面に出すメッセージの頭に付ける名前（どの読み込みで除いたかが分かるように）。
def keep_last_runs(df: pd.DataFrame, who: str) -> pd.DataFrame:
    """条件ごとに最後に書かれた実行の行だけを残す（回し直した実行の途中までの行を捨てる）。"""
    last_run = df.groupby(CONDITION_KEYS, dropna=False)["run_id"].transform("last")
    stale = df["run_id"] != last_run
    if stale.any():
        dropped = sorted(df.loc[stale, "run_id"].unique())
        print(f"[{who}] 回し直された条件の古い実行 {len(dropped)} 件を除外: {dropped}")
    return df[~stale]


# 【load_metrics の説明】
#   detection_metrics.csv を読み込み，使いやすい形に整えて返す。
#     pd.read_csv(path)   : CSV ファイルを表（DataFrame）として読み込む。
#     df[col] = np.nan    : 列が無ければ，全部が空（NaN）の列を足す（古いログでも動くようにするため）。
#
#     keep_last_runs(df, ...) : 回し直した条件は，最後の実行の行だけを残す（上の【keep_last_runs の説明】を参照）。
#     zip(A, B)           : 2つの列から，同じ位置の値を1組ずつ取り出す。
#     df.sort_values(...) : 指定した列の順に，行を並べ替える。
def load_metrics(path: str) -> pd.DataFrame:
    """
    detection_metrics.csv を読み，作図用のラベル列を付けて返す。

    同じ条件のログが複数回ぶん入っている場合（失敗して回し直したときなど，
    ログは追記されるので途中までの行が残る）は，最後に書かれた実行だけを使う。
    混ぜると，trial の平均に同じ条件が重複して数えられてしまうため。
    """
    df = pd.read_csv(path)
    for col in STRUCTURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = keep_last_runs(df, "load_metrics")

    df["attack"] = [attack_label(t, z) for t, z in zip(df["attack_type"], df["alie_z"])]
    sort_cols = CONDITION_KEYS + ["round", "detector"]
    return df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 2) 図：AUC 対 ラウンド（trial 平均±標準偏差）
# ============================================================
# 【plot_auc_by_round の説明】
#   検知性能が，学習の進み具合（ラウンド）とともにどう変わるかを見る図。論文に載せる体裁で描いている。
#     構成 : 小さなグラフを格子状に並べる。行 = 検知器（D1〜D4），列 = 攻撃（左端が符号反転，その右に ALIE を z の小さい順）。
#            ALIE の z は 8 通りあって列が多くなるので，--round-zs で載せる z を絞った版も作れる（detection_auc_by_round_selected.png）。
#     1つのグラフの中 : 横軸がラウンド，縦軸が AUC。alpha ごとに線を1本ずつ描く（色・線の種類・印を変える）。
#     線           : 10回の trial の平均。
#     薄い帯       : 平均 ± 標準偏差（trial によるばらつきの幅）。
#     点線（0.5）  : でたらめに判定したときの AUC。これより上なら見分けられていて，
#                    下なら「悪性のほうが良性らしい」と判定されている。
#
#   前半は，描くものの準備（どの攻撃・どの alpha があるかを調べ，色や線の種類を決める）。
#     attacked = metrics[metrics["attack_type"] != "none"] : 攻撃なしの行を除く（悪性がいないので AUC が無い）。
#     PAPER_RC / alpha_styles(...) : 図の見た目の設定と，alpha ごとの線の見た目（ファイルの上のほうで定義）。
#   同じ描き方を「4つの検知器をまとめた図」と「検知器ごとの図」の両方で使うので，共通の部分を2つの関数に分けている。
#     _auc_by_round_style(...)   : 攻撃の並び順と表題，alpha ごとの線の見た目を決める。
#     _draw_auc_round_panel(...) : グラフ1つぶん（ある検知器の，ある攻撃）を描く。
#   後半の3重の for で，グラフを1つずつ描いている（検知器 -> 攻撃 -> alpha の順）。
#     cell.groupby("round")["auc"].agg(["mean", "std"])
#         : ラウンドごとに，trial をまたいだ AUC の平均と標準偏差を求める。
#     ax.plot(...)         : 線を描く。 ax.fill_between(...) : 2本の線のあいだを薄く塗る（帯）。
#     .clip(0.0, 1.0)      : 値を 0〜1 の範囲に収める（帯が図の外にはみ出さないように）。
#     fig.savefig(outpath, dpi=300) : 図をファイルに保存する。dpi は解像度（大きいほど細かい）。
def _auc_by_round_style(attacked: pd.DataFrame, only_zs=None):
    """攻撃の並び・表題，alpha→(色,線種,印)の対応を作る（まとめ版・検知器ごと版で共通に使う）。

    符号反転（A1）を左端に置き, その右に ALIE（A2）を z の小さい順に並べる（卒論の表と同じ A1 → A2 の順）。
    only_zs を渡すと, ALIE はその z だけを並べる（論文に載せる列を絞るため）。
    alpha は色（赤・青・緑）に加えて線種とマーカーでも区別し, 白黒印刷でも読めるようにする。
    """
    alie_zs = sorted(attacked.loc[attacked["attack_type"] == "alie", "alie_z"].dropna().unique())
    if only_zs is not None:
        alie_zs = [z for z in alie_zs if any(abs(z - w) < 1e-9 for w in only_zs)]
    alie_attacks = [attack_label("alie", z) for z in alie_zs]
    others = sorted(set(attacked["attack"]) - {a for a in attacked["attack"].unique() if str(a).startswith("alie")})
    attacks = others + alie_attacks
    attack_titles = {"sign_flip": "Sign flip (A1)"}
    for z in alie_zs:
        title = "ALIE (A2)" if len(alie_zs) == 1 else rf"ALIE $z={z:g}$"
        attack_titles[attack_label("alie", z)] = title
    alphas = sorted(attacked["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)
    return attacks, attack_titles, alphas, style


def _draw_auc_round_panel(ax, sub: pd.DataFrame, attack: str, alphas, style) -> None:
    """1つの（検知器，攻撃）の組ぶんのパネルを ax に描く（まとめ版・検知器ごと版で共通）。"""
    for alpha in alphas:
        cell = sub[(sub["alpha"] == alpha) & (sub["attack"] == attack)]
        if cell.empty:
            continue
        agg = cell.groupby("round")["auc"].agg(["mean", "std"]).reset_index()
        color, ls, marker = style[alpha]
        std = agg["std"].fillna(0.0)
        ax.fill_between(
            agg["round"],
            (agg["mean"] - std).clip(0.0, 1.0),
            (agg["mean"] + std).clip(0.0, 1.0),
            color=color, alpha=0.10, lw=0,
        )
        ax.plot(
            agg["round"], agg["mean"], color=color, ls=ls, lw=1.0,
            marker=marker, ms=3.5, markevery=5, mfc="white", mew=0.8,
            label=rf"$\alpha={alpha:g}$",
        )
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", zorder=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])


def plot_auc_by_round(metrics: pd.DataFrame, outpath: str, only_zs=None) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[plot_auc_by_round] 攻撃ありの条件が無いためスキップ")
        return

    # 行 = 検知器, 列 = 攻撃 に分け, 1パネルあたり α の3系列だけにする.
    # only_zs を渡すと ALIE の列をその z だけに絞る（--round-zs で指定）.
    attacks, attack_titles, alphas, style = _auc_by_round_style(attacked, only_zs)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            len(DETECTORS), len(attacks),
            figsize=(1.7 * max(len(attacks), 2), 1.35 * len(DETECTORS) + 0.6),
            sharex=True, sharey=True, squeeze=False,
        )
        for r, det in enumerate(DETECTORS):
            sub = attacked[attacked["detector"] == det]
            for c, attack in enumerate(attacks):
                ax = axes[r, c]
                _draw_auc_round_panel(ax, sub, attack, alphas, style)
                if r == 0:
                    ax.set_title(attack_titles.get(attack, attack))
                if c == 0:
                    ax.set_ylabel(f"{DETECTOR_LABELS[det]}\nAUC")
        for ax in axes[-1]:
            ax.set_xlabel("Round")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(labels),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.6, w_pad=0.6)
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【plot_auc_by_round_by_detector の説明】
#   plot_auc_by_round と中身（線の引き方・色・形）は同じだが，4つの検知器を
#   1枚の画像にまとめず，検知器ごとに別ファイル（1行 x 攻撃の列数）へ保存する。
#     outpath_pattern : ファイル名のひな形。中の "{detector}" が d1_norm 等に置き換わる。
def plot_auc_by_round_by_detector(metrics: pd.DataFrame, outpath_pattern: str) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[plot_auc_by_round_by_detector] 攻撃ありの条件が無いためスキップ")
        return

    attacks, attack_titles, alphas, style = _auc_by_round_style(attacked)

    with plt.rc_context(PAPER_RC):
        for det in DETECTORS:
            sub = attacked[attacked["detector"] == det]
            fig, axes = plt.subplots(
                1, len(attacks),
                figsize=(1.7 * max(len(attacks), 2), 2.1),
                sharex=True, sharey=True, squeeze=False,
            )
            for c, attack in enumerate(attacks):
                ax = axes[0, c]
                _draw_auc_round_panel(ax, sub, attack, alphas, style)
                ax.set_title(attack_titles.get(attack, attack))
                if c == 0:
                    ax.set_ylabel("AUC")
                ax.set_xlabel("Round")
            # fig.legend を別枠で浮かせると，suptitle との間隔調整がタイトごとに
            # ずれて余白が安定しないため，凡例は1つ目のパネルの中に収める。
            axes[0, 0].legend(
                fontsize=6, loc="lower right", frameon=True, framealpha=0.9, handlelength=2.2,
            )
            # tight_layout() をまず余白指定なしで呼び，パネル同士の間隔だけ自動調整させる。
            # そのあと上端だけ subplots_adjust で直接指定し，suptitle 用の余白を必要な分だけ確保する
            # （tight_layout の rect 引数と suptitle の組み合わせは余白量が安定しないため）。
            fig.tight_layout()
            fig.subplots_adjust(top=0.72)
            fig.suptitle(DETECTOR_LABELS[det], y=0.90, fontsize=9)
            outpath = outpath_pattern.format(detector=det)
            fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"[saved] {outpath}")


# ============================================================
# 3) 図：push_ratio 対 AUC（中心仮説の検証。符号反転だけ）
# ============================================================
# 【spearman の説明】
#   2つの量の「順位相関係数」（Spearman の順位相関）を求める。
#   値そのものではなく「大きさの順位」どうしの相関を見るので，関係が直線でなくても，
#   「片方が大きいほど，もう片方も大きい」という傾向があれば 1 に近くなる。
#     +1 に近い : 片方が大きいと，もう片方も大きい。
#     -1 に近い : 片方が大きいと，もう片方は小さい。
#      0 に近い : 関係が見られない。
#   点が3つ未満のとき，またはどちらかの値がすべて同じとき（順位が付けられない）は，計算できないので nan を返す。
#     x.rank()            : 値を順位に置き換える（同じ値どうしは，順位の平均を割り当てる）。
#     a.corr(b)           : 2つの列の相関係数（Pearson）。順位に対して計算すると，Spearman の順位相関になる。
#     x.nunique()         : 値の種類の数。1 なら，すべて同じ値。
def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman の順位相関係数。計算できない場合（点が少ない，値が一定）は nan。"""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


# 【rho_text の説明】
#   順位相関係数を，図に書き込む文字列にする（例: ρ=0.45）。計算できなかった場合（nan）は「ρ: n/a」と書く。
#     np.isfinite(rho) : 値が普通の数値なら True。
def rho_text(rho: float) -> str:
    """順位相関係数を図に書く文字列にする。"""
    return rf"$\rho={rho:.2f}$" if np.isfinite(rho) else r"$\rho$: n/a"


# 【sign_flip_rows の説明】
#   push_ratio の分析に使う行（符号反転で，push_ratio と AUC がどちらも普通の数値である行）だけを取り出す。
#   ALIE を入れない理由は，ファイルの先頭の【push_ratio の分析から ALIE を除いている理由】を参照。
#   下の図（plot_pushratio_vs_auc）と表（pushratio_corr_table）が同じ行を使うように，関数にまとめている。
#     np.isfinite(列) : 値が普通の数値である行に True が付く（空の行や無限大を除く）。
def sign_flip_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """push_ratio の分析対象（符号反転で，push_ratio と AUC が有効な行）を返す。"""
    keep = (
        (metrics["attack_type"] == "sign_flip")
        & np.isfinite(metrics["push_ratio"])
        & np.isfinite(metrics["auc"])
    )
    return metrics[keep].copy()


# 【plot_pushratio_vs_auc の説明】
#   研究の中心仮説を確かめるための散布図。符号反転だけを対象にする。
#     横軸 : push_ratio（悪性が良性の集団からどれだけ外に押し出されているか。1 より小さいと集団の内側）。
#            対数の目盛りにしている（ax.set_xscale("log")）。値の幅が広いので，小さい側も見やすくするため。
#     縦軸 : AUC。
#     点   : 1つの点が「1つの実行の，1つのラウンド」。色と印の形が alpha を表す。
#     構成 : 検知器ごとに1枚，計4枚のグラフを 2×2 に並べる。
#     縦の点線 : push_ratio = 1（悪性が良性の広がりのちょうど縁にいる位置）。
#     横の点線 : AUC = 0.5（でたらめに判定したときの値）。
#     右下の数値 : 順位相関係数（spearman）。all は全部の点，その下は alpha ごと。
#   「push_ratio が小さいほど AUC が下がる」という傾向が見えるかどうかを，この図で確かめる。
#
#   alpha ごとの相関も出している理由
#     符号反転には強さを決める値が無いので，push_ratio を動かしているのは主に alpha である。
#     そのため，全部の点で相関があっても，それは「alpha が違うと AUC が違う」ことを言い換えただけかもしれない。
#     同じ alpha の点だけで見ても push_ratio と AUC が連動していれば，
#     push_ratio そのものが検知のしやすさと結びついている，と言いやすくなる。
#
#   同じ描き方を「4つの検知器をまとめた図」と「検知器ごとの図」の両方で使うので，
#   グラフ1つぶんを描く部分を _draw_pushratio_panel(...) に分けている。
#   以前は ALIE の点も一緒に描いていたため，凡例を「符号反転 / ALIE」の大分類にまとめる工夫をしていたが，
#   対象を符号反転だけにしたので，凡例は alpha の区別だけになった。
#     for ax, det in zip(axes.flat, DETECTORS): : 4枚のグラフと4つの検知器を，1組ずつ対応させて回す。
#     ax.scatter(x, y, ...)   : 点を打つ。 rasterized=True は，点が多くてもファイルが重くならないようにする指定。
#     ax.text(..., transform=ax.transAxes) : グラフの中に文字を書く。位置を「グラフの枠の中での割合」で指定している
#                                            （0.97, 0.05 は右下の隅の近く）。
def _draw_pushratio_panel(ax, sub: pd.DataFrame, alphas, style, text_size: float = 6.5) -> None:
    """1つの検知器ぶんの散布図と順位相関係数を ax に描く（まとめ版・検知器ごと版で共通）。"""
    lines = [f"all: {rho_text(spearman(sub['push_ratio'], sub['auc']))}"]
    for alpha in alphas:
        cell = sub[sub["alpha"] == alpha]
        if cell.empty:
            continue
        color, _, marker = style[alpha]
        ax.scatter(
            cell["push_ratio"], cell["auc"], s=7, color=color, marker=marker,
            alpha=0.45, linewidths=0, rasterized=True, label=rf"$\alpha={alpha:g}$",
        )
        lines.append(rf"$\alpha={alpha:g}$: " + rho_text(spearman(cell["push_ratio"], cell["auc"])))
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", zorder=0)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":", zorder=0)
    ax.text(
        0.97, 0.05, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
        fontsize=text_size, bbox=dict(facecolor="white", edgecolor="black", linewidth=0.4, pad=2.0),
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])


def plot_pushratio_vs_auc(metrics: pd.DataFrame, outpath: str) -> None:
    rows = sign_flip_rows(metrics)
    if rows.empty:
        print("[plot_pushratio_vs_auc] 符号反転で push_ratio が有効な行が無いためスキップ")
        return

    alphas = sorted(rows["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.4), sharex=True, sharey=True)
        for ax, det in zip(axes.flat, DETECTORS):
            _draw_pushratio_panel(ax, rows[rows["detector"] == det], alphas, style)
            ax.set_title(DETECTOR_LABELS[det])
        for ax in axes[-1]:
            ax.set_xlabel("push_ratio (sign flip)")
        for ax in axes[:, 0]:
            ax.set_ylabel("AUC")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = fig.legend(
            handles, labels, loc="upper center", ncol=len(labels),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), markerscale=2.0,
        )
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)  # 凡例の印は，薄くせずはっきり描く
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.6)
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【plot_pushratio_vs_auc_by_detector の説明】
#   plot_pushratio_vs_auc と中身（点の打ち方・色・右下の順位相関係数）は同じだが，4つの検知器を
#   1枚の画像にまとめず，検知器ごとに別ファイルへ保存する。対象は，同じく符号反転だけ。
#     outpath_pattern : ファイル名のひな形。中の "{detector}" が d1_norm 等に置き換わる。
def plot_pushratio_vs_auc_by_detector(metrics: pd.DataFrame, outpath_pattern: str) -> None:
    rows = sign_flip_rows(metrics)
    if rows.empty:
        print("[plot_pushratio_vs_auc_by_detector] 符号反転で push_ratio が有効な行が無いためスキップ")
        return

    alphas = sorted(rows["alpha"].unique(), reverse=True)
    style = alpha_styles(alphas)

    with plt.rc_context(PAPER_RC):
        for det in DETECTORS:
            fig, ax = plt.subplots(figsize=(3.2, 2.7))
            _draw_pushratio_panel(ax, rows[rows["detector"] == det], alphas, style)
            ax.set_title(DETECTOR_LABELS[det])
            ax.set_xlabel("push_ratio (sign flip)")
            ax.set_ylabel("AUC")
            # 凡例は，点に重ならないよう，グラフの枠の外（上）に置く。
            handles, labels = ax.get_legend_handles_labels()
            legend = fig.legend(
                handles, labels, loc="upper center", ncol=len(labels),
                frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
                bbox_to_anchor=(0.5, 1.0), markerscale=2.0,
            )
            for handle in legend.legend_handles:
                handle.set_alpha(1.0)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            outpath = outpath_pattern.format(detector=det)
            fig.savefig(outpath, dpi=300)
            plt.close(fig)
            print(f"[saved] {outpath}")


# 【pushratio_corr_table の説明】
#   上の図の右下に書いた順位相関係数を，表にして保存する。符号反転だけを対象にする。
#     行 : alpha（1行目の all は，全部の alpha をまとめたもの）。
#     列 : n_points（相関の計算に使った点の数。1点 = 1つの実行の，1つのラウンド）と，検知器ごとの順位相関係数。
#   all の行の相関が高くても，alpha ごとの行の相関が低ければ，
#   push_ratio と AUC の関係は alpha の違いで説明できてしまう，ということになる（上の図の説明を参照）。
#   nan は，計算できなかったことを表す（AUC がすべて同じ値だった場合など）。
#     groups = [("all", rows)] + [...] : 「名前」と「その名前で使う行」の組を並べたリスト。先頭が全体，続いて alpha ごと。
#     records.append({...})            : 表の1行ぶんを，辞書の形でリストに足していく。
#     pd.DataFrame(records)            : 辞書のリストから，表を作る。
def pushratio_corr_table(metrics: pd.DataFrame, outpath: str) -> None:
    """符号反転での push_ratio と AUC の Spearman 順位相関を，全体と alpha ごとに求めて保存する。"""
    rows = sign_flip_rows(metrics)
    if rows.empty:
        print("[pushratio_corr_table] 符号反転で push_ratio が有効な行が無いためスキップ")
        return
    alphas = sorted(rows["alpha"].unique(), reverse=True)
    groups = [("all", rows)] + [(f"{a:g}", rows[rows["alpha"] == a]) for a in alphas]
    records = []
    for name, group in groups:
        record = {"alpha": name, "n_points": int((group["detector"] == DETECTORS[0]).sum())}
        for det in DETECTORS:
            cell = group[group["detector"] == det]
            record[det] = spearman(cell["push_ratio"], cell["auc"])
        records.append(record)
    table = pd.DataFrame(records).set_index("alpha").round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print("\n=== 符号反転：push_ratio と AUC の順位相関（Spearman） ===")
    print(table.to_string())


# ============================================================
# 4) 図：z 対 AUC（ALIE だけ）
# ============================================================
# 【plot_auc_by_z の説明】
#   ALIE の強さ z を変えたときに，検知性能がどう変わるかを見る図。ALIE だけを対象にする。
#   ALIE では push_ratio がほぼ z に等しいので，push_ratio の代わりに z を横軸にしている。
#     構成 : 検知器ごとに1枚，計4枚のグラフを横に並べる。
#     横軸 : z。 縦軸 : AUC。alpha ごとに線を1本ずつ描く（見た目は「AUC 対 ラウンド」の図と同じ）。
#     点   : 10回の trial の平均。
#     縦の棒 : 平均 ± 標準偏差（trial によるばらつきの幅）。
#     点線（0.5） : でたらめに判定したときの AUC。0 に近い場合は，「見分けられない」のではなく，
#                   「悪性のほうが良性よりも普通に見えている」（判定が逆向きにそろっている）ことを表す。
#   同じ z で alpha によって線が離れていれば，それは push_ratio では説明できない違いである
#   （ALIE では，alpha を変えても push_ratio は変わらないため）。
#
#   平均の取り方（2段階）
#     1. まず，実行（trial）ごとに，最後の last_k ラウンドの AUC を平均して，1つの値にする。
#     2. 次に，その値について，trial をまたいだ平均と標準偏差を求める。
#   同じ実行の中のラウンドどうしは似た値になるので，ラウンドと trial をまとめて標準偏差を取ると，
#   ばらつきを小さく見積もってしまう。そのため，先に実行ごとの値にまとめている。
#     tail.groupby([...])["auc"].mean() : （検知器，alpha，z，trial）の組ごとに，最後の数ラウンドの平均を取る（手順1）。
#     per_trial.groupby([...])["auc"].agg(["mean", "std"]) : trial をまたいだ平均と標準偏差を求める（手順2）。
#     ax.errorbar(x, y, yerr=...) : 線と点に加えて，上下に誤差の棒を描く。capsize は棒の先の横線の長さ。
def plot_auc_by_z(metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    alie = metrics[(metrics["attack_type"] == "alie") & metrics["alie_z"].notna()].copy()
    if alie.empty:
        print("[plot_auc_by_z] ALIE の条件が無いためスキップ")
        return

    max_round = alie["round"].max()
    tail = alie[alie["round"] > max_round - last_k]
    per_trial = tail.groupby(["detector", "alpha", "alie_z", "trial"])["auc"].mean().reset_index()
    summary = per_trial.groupby(["detector", "alpha", "alie_z"])["auc"].agg(["mean", "std"]).reset_index()

    alphas = sorted(alie["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(DETECTORS), figsize=(1.75 * len(DETECTORS), 2.0), sharex=True, sharey=True, squeeze=False,
        )
        for ax, det in zip(axes.flat, DETECTORS):
            for alpha in alphas:
                cell = summary[(summary["detector"] == det) & (summary["alpha"] == alpha)].sort_values("alie_z")
                if cell.empty:
                    continue
                color, ls, marker = style[alpha]
                ax.errorbar(
                    cell["alie_z"], cell["mean"], yerr=cell["std"].fillna(0.0), color=color, ls=ls, lw=1.0,
                    marker=marker, ms=3.5, mfc="white", mew=0.8, elinewidth=0.6, capsize=1.5,
                    label=rf"$\alpha={alpha:g}$",
                )
            ax.axhline(0.5, color="gray", lw=0.5, ls=":", zorder=0)
            ax.set_title(DETECTOR_LABELS[det])
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_xlabel(r"ALIE $z$")
        axes[0, 0].set_ylabel("AUC")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(labels),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=0.6)
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# ============================================================
# 5) 最終ラウンド付近の平均 AUC 一覧
# ============================================================
# 【final_auc_table の説明】
#   結果を数値にまとめた表。行が（alpha，攻撃），列が検知器ごとの「平均」と「標準偏差」。
#   「最後の last_k ラウンド」（既定は5ラウンド）での検知性能を，次の2段階でまとめる
#   （plot_auc_by_z と同じ方法）。
#     1. 実行（trial）ごとに，最後の last_k ラウンドの AUC を平均して，1つの値にする。
#     2. その値について，trial をまたいだ平均（mean）と標準偏差（std）を求める。
#   同じ実行の中のラウンドどうしは似た値になるので，ラウンドと trial をまとめて標準偏差を取ると，
#   ばらつきを小さく見積もってしまう。そのため，先に実行ごとの値にまとめている。
#   平均の値そのものは，ラウンドと trial をまとめて平均した場合と同じになる（どの trial もラウンド数が同じため）。
#   最後の数ラウンドを使うのは，学習がある程度進んだ段階での検知性能を見るため。
#     max_round = attacked["round"].max()                : 最後のラウンドの番号。
#     tail = attacked[attacked["round"] > max_round - last_k] : 最後の last_k ラウンドの行だけを取り出す
#                                                              （50 ラウンドなら 46〜50）。
#     .groupby([...])["auc"].mean() : （alpha，攻撃，検知器，trial）の組ごとに，AUC の平均を取る（手順1）。
#     .agg(["mean", "std"])         : trial をまたいだ平均と標準偏差を求める（手順2）。
#     .unstack("detector")          : 検知器を列の方向に並べる。列の名前は「d1_norm_mean」「d1_norm_std」の形に直す。
#     .round(3)                     : 小数第3位までに丸める。
#     table.to_csv(outpath)         : 表を CSV ファイルに保存する。
def final_auc_table(metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[final_auc_table] 攻撃ありの条件が無いためスキップ")
        return
    max_round = attacked["round"].max()
    tail = attacked[attacked["round"] > max_round - last_k]
    per_trial = tail.groupby(["alpha", "attack", "detector", "trial"])["auc"].mean()
    summary = per_trial.groupby(["alpha", "attack", "detector"]).agg(["mean", "std"]).unstack("detector")
    summary.columns = [f"{det}_{stat}" for stat, det in summary.columns]
    columns = [f"{det}_{stat}" for det in DETECTORS for stat in ("mean", "std")]
    table = summary[[c for c in columns if c in summary.columns]].round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 最終{last_k}ラウンドの AUC（trial ごとに平均してから，trial 間の平均と標準偏差） ===")
    print(table.to_string())


# ============================================================
# 6) 分布構造の診断指標と AUC の対応表
# ============================================================
# 【structure_table の説明】
#   考察のための表。（alpha，攻撃）ごとに，説明する側の指標（spread，push_ratio，分布構造の診断指標）と，
#   結果の側の指標（検知器ごとの AUC）を，横に並べる。
#   「push_ratio はほぼ同じなのに AUC が違う」という条件があったとき，
#   どの診断指標が一緒に動いているかを見て，理由を考える材料にする。
#   攻撃なしの条件も入れている（良性だけのときの基準値になる）。
#     metrics[STRUCTURE_COLS].isna().all().all() : 診断指標の列が全部空なら True（その場合は表を作らない）。
#     per_round = tail[tail["detector"] == DETECTORS[0]]
#         : 説明する側の指標は，同じラウンドならどの検知器の行でも同じ値なので，
#           重複して数えないよう，検知器1つぶんの行だけを使う。
#     .unstack("detector")           : 検知器を，列の方向に並べ直す。
#     .rename(columns=lambda d: f"auc_{d}") : 列の名前の頭に auc_ を付ける（auc_d1_norm など）。
#     explanatory.join(auc)          : 2つの表を，同じ（alpha，攻撃）の行どうしで横につなげる。
def structure_table(metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    """
    (alpha, 攻撃) ごとに，最終 last_k ラウンドの spread / push_ratio / 構造指標と
    検知器別 AUC を並べる。攻撃なし条件も含める（正常更新だけの pc1_ratio の基準値）。
    push_ratio がほぼ同じ条件間で AUC が変わるとき，構造指標が一緒に動いているかを見る。
    """
    if metrics[STRUCTURE_COLS].isna().all().all():
        print("[structure_table] 構造指標の列が無いためスキップ")
        return
    max_round = metrics["round"].max()
    tail = metrics[metrics["round"] > max_round - last_k]
    # 説明変数は検知器によらず同じ値なので，1検知器分だけ取り出して平均する
    per_round = tail[tail["detector"] == DETECTORS[0]]
    explanatory = per_round.groupby(["alpha", "attack"])[
        ["benign_spread", "push_ratio"] + STRUCTURE_COLS
    ].mean()
    auc = (
        tail.groupby(["alpha", "attack", "detector"])["auc"].mean().unstack("detector")
        .rename(columns=lambda d: f"auc_{d}")
    )
    table = explanatory.join(auc).round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 最終{last_k}ラウンド平均の構造指標と AUC ===")
    print(table.to_string())


# ============================================================
# 6b) ラウンドの区間ごとの AUC（学習の段階による変化を数値で示す表）
# ============================================================
# 【segment_auc_table の説明】
#   AUC 対ラウンドの図を，数値の表にしたもの。ラウンドを SEGMENTS の区間に分け，
#   （検知器，攻撃，alpha，区間）ごとに AUC を平均する。
#   図だけでは「序盤は 0.3 くらい」のような目分量しかできないので，本文に区間の平均を書くときの根拠にする。
#   平均の取り方は，trial ごとに区間内のラウンドを平均してから，trial をまたいで平均する
#   （どの trial もラウンド数が同じなので，ラウンドと trial をまとめて平均した場合と同じ値になる）。
#     SEGMENTS : 区間の一覧。（最初のラウンド，最後のラウンド）の組。50 ラウンドの実験に合わせている。
#     列の名前 : 「a1000_r1-5」のように，alpha と区間を並べたもの。
SEGMENTS = [(1, 5), (6, 15), (16, 30), (31, 50)]


def segment_auc_table(metrics: pd.DataFrame, outpath: str) -> None:
    """ラウンドの区間ごとの AUC（trial 平均）を，検知器 × 攻撃 × alpha × 区間で保存する。"""
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[segment_auc_table] 攻撃ありの条件が無いためスキップ")
        return
    pieces = []
    for lo, hi in SEGMENTS:
        cell = attacked[(attacked["round"] >= lo) & (attacked["round"] <= hi)]
        if cell.empty:
            continue
        per_trial = cell.groupby(["detector", "attack", "alpha", "trial"])["auc"].mean()
        seg = per_trial.groupby(["detector", "attack", "alpha"]).mean().rename(f"r{lo}-{hi}")
        pieces.append(seg)
    table = pd.concat(pieces, axis=1).unstack("alpha")
    table.columns = [f"a{alpha:g}_{seg}" for seg, alpha in table.columns]
    alphas = sorted(attacked["alpha"].unique(), reverse=True)
    order = [f"a{a:g}_r{lo}-{hi}" for a in alphas for lo, hi in SEGMENTS if f"a{a:g}_r{lo}-{hi}" in table.columns]
    table = table[order].round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")


# 【segment_pushratio_table の説明】
#   segment_auc_table と同じ区間（SEGMENTS）で，push_ratio を平均した表。
#   push_ratio は同じラウンドならどの検知器の行でも同じ値なので，検知器 1 つぶんの行だけを使う。
#   符号反転では学習の段階によって push_ratio の大小関係が入れ替わること，
#   ALIE ではどの区間でも push_ratio がほぼ z になることを，数値で確かめるための表。
def segment_pushratio_table(metrics: pd.DataFrame, outpath: str) -> None:
    """ラウンドの区間ごとの push_ratio（trial 平均）を，攻撃 × alpha × 区間で保存する。"""
    rows = metrics[(metrics["attack_type"] != "none") & (metrics["detector"] == DETECTORS[0])].copy()
    if rows.empty:
        print("[segment_pushratio_table] 攻撃ありの条件が無いためスキップ")
        return
    pieces = []
    for lo, hi in SEGMENTS:
        cell = rows[(rows["round"] >= lo) & (rows["round"] <= hi)]
        if cell.empty:
            continue
        per_trial = cell.groupby(["attack", "alpha", "trial"])["push_ratio"].mean()
        pieces.append(per_trial.groupby(["attack", "alpha"]).mean().rename(f"r{lo}-{hi}"))
    table = pd.concat(pieces, axis=1).unstack("alpha")
    table.columns = [f"a{alpha:g}_{seg}" for seg, alpha in table.columns]
    alphas = sorted(rows["alpha"].unique(), reverse=True)
    order = [f"a{a:g}_r{lo}-{hi}" for a in alphas for lo, hi in SEGMENTS if f"a{a:g}_r{lo}-{hi}" in table.columns]
    table = table[order].round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")


# ============================================================
# 7) 図：テスト精度のラウンド推移 / 表：最終精度と収束の確認（results.csv があるときだけ）
# ============================================================
# 【load_results の説明】
#   results.csv（ラウンドごとのグローバルモデルのテスト精度）を読む。
#   detection_metrics.csv と同じく，回し直した条件は最後の実行だけを残す（keep_last_runs）。
#   round 0 は，学習を始める前（初期のモデル）の精度である。
def load_results(path: str) -> pd.DataFrame:
    """results.csv を読み，回し直した条件は最後の実行だけを残して返す。"""
    df = keep_last_runs(pd.read_csv(path), "load_results")
    df = df.copy()
    df["attack"] = [attack_label(t, z) for t, z in zip(df["attack_type"], df["alie_z"])]
    return df.sort_values(CONDITION_KEYS + ["round"]).reset_index(drop=True)


# 【plot_accuracy_by_round の説明】
#   学習の進み具合と，攻撃による精度の低下を見る図。
#     構成 : alpha ごとに1枚，計3枚のグラフを横に並べる。
#     横軸 : ラウンド。 縦軸 : テスト精度（%）。
#     線   : 攻撃なし（黒の実線），符号反転（赤の破線），ALIE の z=1 と z=3（青と緑）。
#            ALIE の z は8通りあるが，全部描くと線が重なって読めないので，代表として2つだけ描く。
#            すべての z の値は accuracy_table.csv にある。
#     薄い帯 : 平均 ± 標準偏差（trial によるばらつきの幅）。
#     縦軸の範囲 : 序盤の低い精度まで入れると終盤の差が見えないので，ylim で下限を決めている（既定は 60%）。
#     ACC_LINES : 描く線の一覧。（攻撃の名前，図での名前，色，線の種類）。
ACC_LINES = [
    ("none", "No attack", "black", "-"),
    ("sign_flip", "Sign flip (A1)", "tab:red", "--"),
    ("alie_z1", r"ALIE $z=1$", "tab:blue", "-."),
    ("alie_z3", r"ALIE $z=3$", "tab:green", ":"),
]


def plot_accuracy_by_round(results: pd.DataFrame, outpath: str, ymin: float = 60.0) -> None:
    alphas = sorted(results["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(alphas), figsize=(2.2 * len(alphas), 2.2), sharey=True, squeeze=False)
        for ax, alpha in zip(axes.flat, alphas):
            for attack, label, color, ls in ACC_LINES:
                cell = results[(results["alpha"] == alpha) & (results["attack"] == attack)]
                if cell.empty:
                    continue
                agg = (cell.groupby("round")["accuracy"].agg(["mean", "std"]) * 100).reset_index()
                std = agg["std"].fillna(0.0)
                ax.fill_between(agg["round"], agg["mean"] - std, agg["mean"] + std, color=color, alpha=0.10, lw=0)
                ax.plot(agg["round"], agg["mean"], color=color, ls=ls, lw=1.0, label=label)
            ax.set_title(rf"$\alpha={alpha:g}$")
            ax.set_xlabel("Round")
            ax.set_ylim(ymin, 100.0)
        axes[0, 0].set_ylabel("Test accuracy (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(labels),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6)
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【accuracy_table の説明】
#   最終ラウンドの精度と，学習が収束しているかの目安をまとめた表。行が（alpha，攻撃）。
#     final_mean / final_std : 最終ラウンドの精度（%）の，trial 間の平均と標準偏差。
#     final_min              : 最終ラウンドの精度の，trial の中で一番低い値（%）。
#     gain_last_k            : 最後の last_k ラウンドで精度が何ポイント上がったか
#                              （最終ラウンドの精度 − last_k ラウンド前の精度。trial 間の平均）。
#                              0 に近ければ，精度はほぼ頭打ち（収束している）とみなせる。
#                              50 ラウンド・last_k=5 なら，45 ラウンドから 50 ラウンドまでの上がり幅。
#     .pivot(index="trial", columns="round", values="accuracy") : trial を行，ラウンドを列にした表に組み替える。
def accuracy_table(results: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    max_round = int(results["round"].max())
    records = []
    for (alpha, attack), cell in results.groupby(["alpha", "attack"]):
        wide = cell.pivot(index="trial", columns="round", values="accuracy") * 100
        final = wide[max_round]
        gain = final - wide[max_round - last_k]
        records.append({
            "alpha": alpha, "attack": attack,
            "final_mean": final.mean(), "final_std": final.std(), "final_min": final.min(),
            f"gain_last_{last_k}": gain.mean(),
        })
    table = pd.DataFrame(records).set_index(["alpha", "attack"]).round(2)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 最終ラウンド（{max_round}）の精度 [%] と，最後の {last_k} ラウンドでの上がり幅 [ポイント] ===")
    print(table.to_string())


# ============================================================
# 8) 符号反転で悪性と正常が分かれる様子（detection_updates.csv があるときだけ）
# ============================================================
# 【load_updates の説明】
#   detection_updates.csv（1クライアント1行。ラウンドごとの各クライアントの検知スコア）を読む。
#   ここで使うのは符号反転・ALIE・攻撃なしの行（攻撃なしの行は，ノルムの比を比べるときの対照に使う）。
#   detection_metrics.csv と同じく，回し直した条件は最後の実行だけを残す（keep_last_runs）。
#     usecols=... : 必要な列だけを読む（ファイルが大きいので，読み込みを軽くするため）。
UPDATES_COLS = CONDITION_KEYS + ["run_id", "round", "partition_id", "is_malicious", "delta_norm", "d1_norm", "d2_cosine", "d4_spectral"]


def load_updates(path: str) -> pd.DataFrame:
    """detection_updates.csv を読み，符号反転・ALIE・攻撃なしの行を，回し直した条件は最後の実行だけにして返す。"""
    df = keep_last_runs(pd.read_csv(path, usecols=UPDATES_COLS), "load_updates")
    return df[df["attack_type"].isin(["sign_flip", "alie", "none"])].reset_index(drop=True)


# 【_d2_by_group の説明】
#   符号反転の行から，（alpha，trial，ラウンド）ごとに，正常更新と悪性更新それぞれの D2 スコアの平均を求める。
#   列 benign / malicious を持つ表を返す（図と表の両方で使う）。
#     .unstack("is_malicious") : is_malicious（0 か 1）を列の方向に並べる。
def _d2_by_group(updates: pd.DataFrame) -> pd.DataFrame:
    g = updates.groupby(["alpha", "trial", "round", "is_malicious"])["d2_cosine"].mean().unstack("is_malicious")
    return g.rename(columns={0: "benign", 1: "malicious"}).reset_index()


# 【plot_signflip_d2_score_by_round の説明】
#   符号反転で D2 がなぜ序盤から悪性を見分けられるのか，学習が進むとどうなるのかを見る図。
#     横軸 : ラウンド。 縦軸 : D2 スコア（1 - cos。参照方向とのコサイン距離）。
#     線   : alpha ごとに，正常更新と悪性更新の2本ずつ（trial 平均）。薄い帯は ± 標準偏差。
#            alpha は他の図と同じ色・線の種類・印（alpha_styles）。
#            正常更新は塗りつぶした印，悪性更新は白抜きの印で区別する。
#     点線（1.0）: cos = 0 の位置。正常更新と参照方向のなす角を θ とすると，正常は 1 - cos θ，
#                  符号を反転した悪性は 1 + cos θ になるので，正常が cos θ > 0 である限り，
#                  正常は 1 より下，悪性は 1 より上に分かれる。
#     凡例 : alpha の3つ。正常・悪性の区別は，図の中に直接書く（上の線が悪性，下の線が正常）。
def plot_signflip_d2_score_by_round(updates: pd.DataFrame, outpath: str) -> None:
    updates = updates[updates["attack_type"] == "sign_flip"]
    if updates.empty:
        print("[plot_signflip_d2_score_by_round] 符号反転の行が無いためスキップ")
        return
    per_trial = _d2_by_group(updates)
    alphas = sorted(per_trial["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        for alpha in alphas:
            color, ls, marker = style[alpha]
            cell = per_trial[per_trial["alpha"] == alpha]
            for group, mfc in (("benign", color), ("malicious", "white")):
                agg = cell.groupby("round")[group].agg(["mean", "std"]).reset_index()
                std = agg["std"].fillna(0.0)
                ax.fill_between(agg["round"], agg["mean"] - std, agg["mean"] + std, color=color, alpha=0.10, lw=0)
                ax.plot(
                    agg["round"], agg["mean"], color=color, ls=ls, lw=1.0,
                    marker=marker, ms=3.5, markevery=5, mfc=mfc, mew=0.8,
                )
        ax.axhline(1.0, color="gray", lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Round")
        ax.set_ylabel(r"D2 score ($1-\cos$)")
        ax.set_ylim(0.0, 2.0)
        ax.text(0.98, 0.95, "Malicious (open markers)", transform=ax.transAxes, ha="right", va="top")
        ax.text(0.98, 0.05, "Benign (filled markers)", transform=ax.transAxes, ha="right", va="bottom")
        handles = [
            plt.Line2D([], [], color=style[a][0], ls=style[a][1], marker=style[a][2], ms=3.5,
                       mfc="white", mew=0.8, lw=1.0, label=rf"$\alpha={a:g}$")
            for a in alphas
        ]
        fig.legend(
            handles=handles, loc="upper center", ncol=len(handles),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【signflip_separation_table の説明】
#   符号反転で悪性と正常が分かれる様子を，ラウンドの区間ごとに数値で示す表。行が alpha。
#   区間は，学習の序盤（第1-5ラウンド）と，最後の last_k ラウンド（final_auc_table と同じ）。
#     d2_benign / d2_malicious : 正常更新・悪性更新の D2 スコアの平均。
#     perfect_sep              : ラウンドの中で，すべての悪性の D2 スコアが，すべての正常のスコアより
#                                大きかった（完全に分かれた）ラウンドの割合。
#     mal_cos_pc1_all          : 悪性方向（悪性の平均 - 正常の平均）と全更新の第1主成分の |cos|
#                                （detection_metrics.csv の値。D3 の特徴量の PC1 と同じ向き）。
#   平均の取り方は，trial ごとに区間内のラウンドを平均してから，trial をまたいで平均する。
#     sep = ...groupby([...]).apply(...) : （alpha，trial，ラウンド）ごとに「完全に分かれたか」を True/False で求める。
def signflip_separation_table(updates: pd.DataFrame, metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    updates = updates[updates["attack_type"] == "sign_flip"]
    if updates.empty:
        print("[signflip_separation_table] 符号反転の行が無いためスキップ")
        return
    per_round = _d2_by_group(updates)
    sep = (
        updates.groupby(["alpha", "trial", "round"])
        .apply(lambda t: t.loc[t["is_malicious"] == 1, "d2_cosine"].min() > t.loc[t["is_malicious"] == 0, "d2_cosine"].max())
        .rename("perfect_sep").reset_index()
    )
    per_round = per_round.merge(sep, on=["alpha", "trial", "round"])
    pc1 = metrics[(metrics["attack_type"] == "sign_flip") & (metrics["detector"] == DETECTORS[0])]
    per_round = per_round.merge(pc1[["alpha", "trial", "round", "mal_cos_pc1_all"]], on=["alpha", "trial", "round"], how="left")

    max_round = per_round["round"].max()
    segments = [(1, 5), (max_round - last_k + 1, max_round)]
    cols = ["benign", "malicious", "perfect_sep", "mal_cos_pc1_all"]
    names = {"benign": "d2_benign", "malicious": "d2_malicious"}
    pieces = []
    for lo, hi in segments:
        cell = per_round[(per_round["round"] >= lo) & (per_round["round"] <= hi)].astype({"perfect_sep": float})
        seg = cell.groupby(["alpha", "trial"])[cols].mean().groupby("alpha").mean()
        seg.columns = [f"r{lo}-{hi}_{names.get(c, c)}" for c in cols]
        pieces.append(seg)
    table = pd.concat(pieces, axis=1).sort_index(ascending=False).round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print("\n=== 符号反転：D2 スコアの平均，完全に分かれたラウンドの割合，悪性方向と第1主成分の |cos| ===")
    print(table.T.to_string())


# 【plot_signflip_norm_ratio_by_round の説明】
#   符号反転で D1 が学習の途中から悪性を見分けられるようになる理由を見る図。
#     縦軸 : 悪性更新のノルムの平均 ÷ 正常更新のノルムの平均（ラウンドごと。1 なら大きさが揃っている）。
#     線   : alpha ごとに2本ずつ（trial 平均。薄い帯は ± 標準偏差）。
#            白抜きの印 = 符号反転の実行。
#            塗りつぶした印 = 攻撃なしの実行で，同じ trial で符号反転のときに悪性になる4台を「悪性」とみなしたもの（対照）。
#            悪性クライアントの選び方は trial ごとに固定なので，同じ trial の攻撃なしの実行にそのまま当てはめられる。
#            攻撃なしで比が 1 から離れないなら，ノルムの差は4台のデータの性質ではなく攻撃によって生じたことになる。
#     mal_ids : （trial，partition_id）の組で，符号反転のときに悪性だったクライアント。
def _norm_ratio_by_round(rows: pd.DataFrame, mal_col: str) -> pd.DataFrame:
    g = rows.groupby(["alpha", "trial", "round", mal_col])["delta_norm"].mean().unstack(mal_col)
    return (g[1] / g[0]).rename("ratio").reset_index()


def plot_signflip_norm_ratio_by_round(updates: pd.DataFrame, outpath: str) -> None:
    sf = updates[updates["attack_type"] == "sign_flip"]
    none = updates[updates["attack_type"] == "none"]
    if sf.empty or none.empty:
        print("[plot_signflip_norm_ratio_by_round] 符号反転か攻撃なしの行が無いためスキップ")
        return
    mal_ids = sf.loc[sf["is_malicious"] == 1, ["trial", "partition_id"]].drop_duplicates().assign(same_clients=1)
    none = none.merge(mal_ids, on=["trial", "partition_id"], how="left").fillna({"same_clients": 0})
    series = [
        (_norm_ratio_by_round(sf, "is_malicious"), "open"),
        (_norm_ratio_by_round(none, "same_clients"), "filled"),
    ]
    alphas = sorted(sf["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        for alpha in alphas:
            color, ls, marker = style[alpha]
            for ratio, fill in series:
                agg = ratio[ratio["alpha"] == alpha].groupby("round")["ratio"].agg(["mean", "std"]).reset_index()
                std = agg["std"].fillna(0.0)
                ax.fill_between(agg["round"], agg["mean"] - std, agg["mean"] + std, color=color, alpha=0.10, lw=0)
                ax.plot(
                    agg["round"], agg["mean"], color=color, ls=ls, lw=1.0,
                    marker=marker, ms=3.5, markevery=5, mfc="white" if fill == "open" else color, mew=0.8,
                )
        ax.axhline(1.0, color="gray", lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Round")
        ax.set_ylabel("Norm ratio (malicious / benign)")
        # 攻撃なしの線は 1 付近に集まるので，下に余白を取り，その注記を線の下に置く
        ax.set_ylim(0.55, None)
        ax.text(0.02, 0.95, "Sign flip (open markers)", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.98, 0.03, "No attack, same clients (filled markers)", transform=ax.transAxes, ha="right", va="bottom")
        handles = [
            plt.Line2D([], [], color=style[a][0], ls=style[a][1], marker=style[a][2], ms=3.5,
                       mfc="white", mew=0.8, lw=1.0, label=rf"$\alpha={a:g}$")
            for a in alphas
        ]
        fig.legend(
            handles=handles, loc="upper center", ncol=len(handles),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【signflip_norm_dispersion_table の説明】
#   符号反転で，alpha=1 のときに D1 と D4 の AUC が下がる理由を見る表。行が alpha。
#   どちらも「正常更新のノルムのばらつき」によって，正常更新が悪性と取り違えられることを示すための表である。
#   最後の last_k ラウンドで，次の量を求める（trial ごとに平均してから，trial をまたいで平均する）。
#     norm_ratio          : 悪性更新のノルムの平均 ÷ 正常更新のノルムの平均。
#     cv_benign           : 正常更新のノルムの変動係数（ラウンド内の 標準偏差 ÷ 平均）。
#     cv_benign_none      : 攻撃なしの実行での，全クライアントのノルムの変動係数（対照）。
#     d1_misranked / d4_misranked
#                         : （正常1台，悪性1台）の組のうち，その検知器が正常のほうを疑わしいと判定した組の割合。
#                           同点は半分と数えるので，1 - AUC に等しい。
#     d1_misranked_below  : D1 の誤った組のうち，その正常更新のノルムが中央値より小さかったものの割合。
#                           D1 は |ノルム - 中央値| をスコアにするので，小さすぎる正常更新も疑わしく見える。
#     d4_misranked_above  : D4 の誤った組のうち，その正常更新のノルムが中央値より大きかったものの割合。
#                           D4 は射影の2乗をスコアにするので，共通の方向に沿って遠くにいる（ノルムの大きい）正常更新が疑わしく見える。
#     （内訳の2列だけは，全 trial・全ラウンドの組をまとめて割合を求める。誤った組が無い alpha は空欄になる）
#     d4_benign_norm_corr : 正常更新どうしでの，D4 のスコアとノルムの順位相関（ラウンドごとに求めて平均）。
#   _misranked(t, score) : 1ラウンドぶんの行 t から，検知器 score で誤った組の数（全体と，中央値より下の正常によるもの）と組の総数を返す。
#     s[:, None] > m[None, :] : 正常の各スコアと悪性の各スコアを総当たりで比べた表（正常の数 × 悪性の数）。
def _misranked(t: pd.DataFrame, score: str) -> pd.Series:
    med = t["delta_norm"].median()
    b = t[t["is_malicious"] == 0]
    m = t.loc[t["is_malicious"] == 1, score].to_numpy()
    s = b[score].to_numpy()
    wrong = (s[:, None] > m[None, :]) + 0.5 * (s[:, None] == m[None, :])
    below = (b["delta_norm"].to_numpy() < med)[:, None]
    return pd.Series({"wrong": wrong.sum(), "wrong_below": (wrong * below).sum(), "pairs": wrong.size})


def signflip_norm_dispersion_table(updates: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    max_round = updates["round"].max()
    tail = updates[updates["round"] > max_round - last_k]
    sf = tail[tail["attack_type"] == "sign_flip"]
    none = tail[tail["attack_type"] == "none"]
    if sf.empty or none.empty:
        print("[signflip_norm_dispersion_table] 符号反転か攻撃なしの行が無いためスキップ")
        return
    keys = ["alpha", "trial", "round"]

    def cv(rows: pd.DataFrame) -> pd.Series:
        g = rows.groupby(keys)["delta_norm"]
        return (g.std() / g.mean()).groupby(["alpha", "trial"]).mean()

    def counts(score: str) -> pd.DataFrame:
        cols = ["delta_norm", score, "is_malicious"]
        return sf.groupby(keys)[cols].apply(_misranked, score=score).groupby(["alpha", "trial"]).sum()

    benign = sf[sf["is_malicious"] == 0]
    d4_corr = (
        benign.groupby(keys)[["d4_spectral", "delta_norm"]]
        .apply(lambda t: t["d4_spectral"].rank().corr(t["delta_norm"].rank()))
        .groupby(["alpha", "trial"]).mean()
    )
    c1, c4 = counts("d1_norm"), counts("d4_spectral")
    per_trial = pd.DataFrame({
        "norm_ratio": _norm_ratio_by_round(sf, "is_malicious").groupby(["alpha", "trial"])["ratio"].mean(),
        "cv_benign": cv(benign),
        "cv_benign_none": cv(none),
        "d1_misranked": c1["wrong"] / c1["pairs"],
        "d4_misranked": c4["wrong"] / c4["pairs"],
        "d4_benign_norm_corr": d4_corr,
    })
    table = per_trial.groupby("alpha").mean()
    # 内訳の割合は，trial ごとに割ると誤った組が少ない trial の値が大きく揺れるので，全 trial の組をまとめて割る
    p1, p4 = c1.groupby("alpha").sum(), c4.groupby("alpha").sum()
    table["d1_misranked_below"] = p1["wrong_below"] / p1["wrong"].where(p1["wrong"] > 0)
    table["d4_misranked_above"] = 1 - p4["wrong_below"] / p4["wrong"].where(p4["wrong"] > 0)
    order = ["norm_ratio", "cv_benign", "cv_benign_none", "d1_misranked", "d1_misranked_below",
             "d4_misranked", "d4_misranked_above", "d4_benign_norm_corr"]
    table = table[order].sort_index(ascending=False).round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 符号反転：最終{last_k}ラウンドのノルムの比，変動係数，D1・D4 が順位を誤った組の割合 ===")
    print(table.to_string())


# 【signflip_norm_cv_by_segment_table の説明】
#   正常更新のノルムのばらつきが，学習の段階とともにどう変わるかを見る表。
#   ラウンドを SEGMENTS の区間に分け，（攻撃，alpha，区間）ごとに正常更新のノルムの変動係数を平均する
#   （ラウンドごとに 標準偏差 ÷ 平均 を求め，trial ごとに区間内で平均してから，trial をまたいで平均する）。
#   符号反転では学習が進むほどばらつきが広がり，攻撃なしでは広がらないことを示すための表。
#   攻撃なしの実行には悪性がいないので，全クライアントを正常として数える。
def signflip_norm_cv_by_segment_table(updates: pd.DataFrame, outpath: str) -> None:
    benign = updates[(updates["is_malicious"] == 0) & updates["attack_type"].isin(["sign_flip", "none"])]
    if benign.empty:
        print("[signflip_norm_cv_by_segment_table] 行が無いためスキップ")
        return
    g = benign.groupby(["attack_type", "alpha", "trial", "round"])["delta_norm"]
    cv = (g.std() / g.mean()).rename("cv").reset_index()
    pieces = []
    for lo, hi in SEGMENTS:
        cell = cv[(cv["round"] >= lo) & (cv["round"] <= hi)]
        seg = cell.groupby(["attack_type", "alpha", "trial"])["cv"].mean().groupby(["attack_type", "alpha"]).mean()
        pieces.append(seg.rename(f"r{lo}-{hi}"))
    table = pd.concat(pieces, axis=1).sort_index(level=[0, 1], ascending=[False, False]).round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print("\n=== 正常更新のノルムの変動係数（ラウンドの区間ごと） ===")
    print(table.to_string())


# 【alie_d2_score_by_z の説明】
#   ALIE で D2 が z <= 1.5 では悪性を「一番正常らしい」と判定し，z = 2〜3 で検知に転じる理由を見る図と表。
#     横軸 : ALIE の z（8 水準）。 縦軸 : 最後の last_k ラウンドの D2 スコア（1 - cos）の平均。
#     線   : alpha ごとに，正常更新（塗りつぶしの印）と悪性更新（白抜きの印）の2本ずつ。
#            trial ごとに最後の last_k ラウンドを平均してから，trial をまたいだ平均（線）と標準偏差（縦棒）を求める。
#     点線（1.0）: cos = 0（参照方向と直角）。
#   悪性の線が正常の線より下にある z では，悪性のほうが参照方向に近く（AUC が 0 に近い），
#   悪性の線が正常の線を越える z で，D2 は検知に転じる。
#   表（alie_d2_score_by_z_table.csv）は，図と同じ平均と標準偏差を，z × alpha × 正常・悪性で並べたもの。
#   表の最後の行（alie_z が "none"）は攻撃なしの実行の値で，全クライアントを正常として数える。
#   攻撃なしでは参照方向はほぼ正常の平均の向きなので，正常の D2 スコアが 1 に近いほど，
#   全員に共通する成分（平均）に比べて各クライアント自身のずれが大きいことを表す。
def _alie_d2_by_z(updates: pd.DataFrame, last_k: int) -> pd.DataFrame:
    alie = updates[updates["attack_type"] == "alie"]
    tail = alie[alie["round"] > alie["round"].max() - last_k]
    per_trial = tail.groupby(["alie_z", "alpha", "is_malicious", "trial"])["d2_cosine"].mean()
    return per_trial.groupby(["alie_z", "alpha", "is_malicious"]).agg(["mean", "std"]).reset_index()


def plot_alie_d2_score_by_z(updates: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    if (updates["attack_type"] == "alie").sum() == 0:
        print("[plot_alie_d2_score_by_z] ALIE の行が無いためスキップ")
        return
    agg = _alie_d2_by_z(updates, last_k)
    alphas = sorted(agg["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        for alpha in alphas:
            color, ls, marker = style[alpha]
            for mal, mfc in ((0, color), (1, "white")):
                cell = agg[(agg["alpha"] == alpha) & (agg["is_malicious"] == mal)].sort_values("alie_z")
                ax.errorbar(
                    cell["alie_z"], cell["mean"], yerr=cell["std"].fillna(0.0),
                    color=color, ls=ls, lw=1.0, marker=marker, ms=3.5, mfc=mfc, mew=0.8,
                    capsize=1.5, elinewidth=0.6,
                )
        ax.axhline(1.0, color="gray", lw=0.5, ls=":", zorder=0)
        ax.set_xlabel(r"ALIE $z$")  # 目盛りは自動（detection_auc_by_z.png と同じ）。0.385 と 0.5 は近くて数字が重なるため
        ax.set_ylabel(r"D2 score ($1-\cos$)")
        ax.set_ylim(0.0, 1.4)
        ax.text(0.02, 0.97, "Benign (filled markers)", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.98, 0.03, "Malicious (open markers)", transform=ax.transAxes, ha="right", va="bottom")
        handles = [
            plt.Line2D([], [], color=style[a][0], ls=style[a][1], marker=style[a][2], ms=3.5,
                       mfc="white", mew=0.8, lw=1.0, label=rf"$\alpha={a:g}$")
            for a in alphas
        ]
        fig.legend(
            handles=handles, loc="upper center", ncol=len(handles),
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


def alie_d2_score_by_z_table(updates: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    if (updates["attack_type"] == "alie").sum() == 0:
        print("[alie_d2_score_by_z_table] ALIE の行が無いためスキップ")
        return
    agg = _alie_d2_by_z(updates, last_k)
    agg["group"] = agg["is_malicious"].map({0: "benign", 1: "malicious"})
    none = updates[updates["attack_type"] == "none"]
    none = none[none["round"] > none["round"].max() - last_k]
    if not none.empty:
        per_trial = none.groupby(["alpha", "trial"])["d2_cosine"].mean()
        none_agg = per_trial.groupby("alpha").agg(["mean", "std"]).reset_index()
        none_agg = none_agg.assign(alie_z="none", is_malicious=0, group="benign")
        agg = pd.concat([agg.assign(alie_z=agg["alie_z"].map(lambda z: f"{z:g}")), none_agg], ignore_index=True)
    table = agg.pivot_table(index="alie_z", columns=["alpha", "group"], values=["mean", "std"], sort=False)
    table.columns = [f"a{alpha:g}_{group}_{stat}" for stat, alpha, group in table.columns]
    alphas = sorted(agg["alpha"].unique(), reverse=True)
    order = [f"a{a:g}_{g}_{st}" for a in alphas for g in ("benign", "malicious") for st in ("mean", "std")]
    table = table[order].round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== ALIE：最終{last_k}ラウンドの D2 スコア（正常・悪性） ===")
    print(table.to_string())


# 【alie_norm_ratio_by_z_table の説明】
#   ALIE で D1 の AUC が z に対して U 字型になる理由を見る表。行が z，列が alpha。
#   値は，最後の last_k ラウンドでの「悪性更新のノルム ÷ 正常更新のノルムの平均」
#   （ラウンドごとに求め，trial ごとに平均してから，trial をまたいで平均する）。
#   1 より小さければ悪性のほうが小さく，1 に近い z で D1 は見分けられなくなる。
def alie_norm_ratio_by_z_table(updates: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    alie = updates[updates["attack_type"] == "alie"]
    if alie.empty:
        print("[alie_norm_ratio_by_z_table] ALIE の行が無いためスキップ")
        return
    tail = alie[alie["round"] > alie["round"].max() - last_k]
    g = tail.groupby(["alie_z", "alpha", "trial", "round", "is_malicious"])["delta_norm"].mean().unstack("is_malicious")
    ratio = (g[1] / g[0]).groupby(["alie_z", "alpha", "trial"]).mean().groupby(["alie_z", "alpha"]).mean()
    table = ratio.unstack("alpha")
    table = table[sorted(table.columns, reverse=True)]
    table.columns = [f"a{a:g}" for a in table.columns]
    table = table.round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== ALIE：最終{last_k}ラウンドの悪性/正常のノルムの比 ===")
    print(table.to_string())


# 【alie_shell の説明】
#   卒論 9.3（高次元の殻）の図と表。ALIE の悪性が z とともに「殻の内側 → 殻の上 → 殻の外」へ動くことを示す。
#   最後の last_k ラウンドで，(alpha, z) ごとに次の量を求める（trial ごとに平均してから，trial をまたいで平均する）。
#     dist_ratio      : push_ratio（悪性と正常の平均との距離 ÷ 正常と平均との距離の平均）。ALIE ではほぼ z。
#     norm_ratio      : 悪性のノルム ÷ 正常のノルムの平均（alie_norm_ratio_by_z_table と同じ値）。
#     benign_norm_cv  : 正常のノルムの変動係数（ラウンド内の 標準偏差 ÷ 平均）。殻の厚みの目安。
#     benign_dist_cv  : 正常の平均からの距離の変動係数。式 push_ratio = z × RMS / Mean から，
#                       RMS / Mean = push_ratio / z = sqrt(1 + CV^2) を使って CV = sqrt((push_ratio / z)^2 - 1) と求める。
#   図（alie_shell_by_z.png）: alpha ごとに 1 枚，計 3 枚を横に並べる。横軸 z，縦軸は「悪性 ÷ 正常」の比。
#     破線 = 距離の比（push_ratio），実線 = ノルムの比。灰色の帯 = 1 ± 正常のノルムの変動係数（正常の殻の厚みの目安）。
#     ノルムの線が帯の中を通る z で，D1 は悪性と正常を見分けにくくなる。
def _alie_shell(updates: pd.DataFrame, metrics: pd.DataFrame, last_k: int) -> pd.DataFrame:
    alie_u = updates[updates["attack_type"] == "alie"]
    tail_u = alie_u[alie_u["round"] > alie_u["round"].max() - last_k]
    keys = ["alie_z", "alpha", "trial"]
    g = tail_u.groupby(keys + ["round", "is_malicious"])["delta_norm"].mean().unstack("is_malicious")
    norm_ratio = (g[1] / g[0]).groupby(keys).mean()
    gb = tail_u[tail_u["is_malicious"] == 0].groupby(keys + ["round"])["delta_norm"]
    norm_cv = (gb.std() / gb.mean()).groupby(keys).mean()
    alie_m = metrics[(metrics["attack_type"] == "alie") & (metrics["detector"] == DETECTORS[0])]
    tail_m = alie_m[alie_m["round"] > alie_m["round"].max() - last_k]
    dist_ratio = tail_m.groupby(keys)["push_ratio"].mean()
    per_trial = pd.DataFrame({"dist_ratio": dist_ratio, "norm_ratio": norm_ratio, "benign_norm_cv": norm_cv})
    table = per_trial.groupby(["alie_z", "alpha"]).mean().reset_index()
    k = (table["dist_ratio"] / table["alie_z"]).clip(lower=1.0)
    table["benign_dist_cv"] = np.sqrt(k ** 2 - 1.0)
    return table


def alie_shell_table(updates: pd.DataFrame, metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    if (updates["attack_type"] == "alie").sum() == 0:
        print("[alie_shell_table] ALIE の行が無いためスキップ")
        return
    table = _alie_shell(updates, metrics, last_k)
    table = table.sort_values(["alpha", "alie_z"], ascending=[False, True]).set_index(["alpha", "alie_z"]).round(3)
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== ALIE：最終{last_k}ラウンドの距離の比・ノルムの比・正常の殻の厚み ===")
    print(table.to_string())


def plot_alie_shell_by_z(updates: pd.DataFrame, metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    if (updates["attack_type"] == "alie").sum() == 0:
        print("[plot_alie_shell_by_z] ALIE の行が無いためスキップ")
        return
    table = _alie_shell(updates, metrics, last_k)
    alphas = sorted(table["alpha"].unique(), reverse=True)  # 1000 → 5 → 1
    style = alpha_styles(alphas)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(alphas), figsize=(2.2 * len(alphas), 2.3), sharey=True, squeeze=False)
        for ax, alpha in zip(axes.flat, alphas):
            color, _, marker = style[alpha]
            cell = table[table["alpha"] == alpha].sort_values("alie_z")
            cv = cell["benign_norm_cv"].mean()
            ax.axhspan(1.0 - cv, 1.0 + cv, color="gray", alpha=0.18, lw=0)
            ax.axhline(1.0, color="gray", lw=0.5, ls=":", zorder=0)
            ax.plot(cell["alie_z"], cell["dist_ratio"], color=color, ls="--", lw=1.0,
                    marker=marker, ms=3.5, mfc="white", mew=0.8, label="Distance from mean (push_ratio)")
            ax.plot(cell["alie_z"], cell["norm_ratio"], color=color, ls="-", lw=1.0,
                    marker=marker, ms=3.5, mfc=color, mew=0.8, label="Norm")
            ax.set_title(rf"$\alpha={alpha:g}$")
            ax.set_xlabel(r"ALIE $z$")
            ax.set_ylim(0.0, 3.2)
        axes[0, 0].set_ylabel("Malicious / benign")
        handles = [
            plt.Line2D([], [], color="black", ls="--", lw=1.0, marker="o", ms=3.5, mfc="white", mew=0.8,
                       label="Distance from mean (push_ratio)"),
            plt.Line2D([], [], color="black", ls="-", lw=1.0, marker="o", ms=3.5, mfc="black", mew=0.8, label="Norm"),
            plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.18, lw=0, label=r"Benign band ($1 \pm$ CV of norm)"),
        ]
        fig.legend(
            handles=handles, loc="upper center", ncol=3,
            frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6)
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")


# ============================================================
# 【main の説明】
#   スクリプトを実行したときに動く本体。
#     ap.add_argument("logdir", ...)  : 名前の前に -- が付かない指定は，必ず書かなければならない（位置で渡す）。
#                                       ここでは，ログのフォルダ。既定値は置いていないので，
#                                       書き忘れると「指定が必要」というエラーになる（別のデータを取り違えないため）。
#     ap.add_argument("--outdir", default=None, ...) : 図と表の保存先。省略できる。
#     ap.add_argument("--last-k", type=int, default=5, ...) : 最後の何ラウンドを平均するか。整数で受け取る。
#     os.path.join(args.logdir, METRICS_CSV_NAME)  : フォルダ名とファイル名をつないで，読み込むファイルのパスを作る。
#     raise SystemExit(メッセージ)    : ファイルが無ければ，メッセージを表示して終了する。
#     outdir = args.outdir or os.path.join(args.logdir, "analysis")
#         : 保存先が指定されていればそれを使い，なければ <ログのフォルダ>/analysis にする。
#   読み込んだあとに，ファイルの場所・行数・実行の数・ラウンドの範囲を表示している。
#   意図したデータを読んでいるかを，ここで確かめられる。
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="検知評価スイープの後処理（集計と作図）")
    ap.add_argument(
        "logdir",
        help=f"{METRICS_CSV_NAME} が入っているフォルダ（例: experiments_detection_v2/_logs）。"
        "取り違えを防ぐため既定値は置かず，毎回指定する。",
    )
    ap.add_argument("--outdir", default=None, help="図と表の保存先（既定: <logdir>/analysis）")
    ap.add_argument("--last-k", type=int, default=5, help="最終平均を取るラウンド数")
    ap.add_argument(
        "--round-zs", default=None,
        help="AUC 対ラウンドの図で，ALIE の列をこの z だけに絞った版も作る（カンマ区切り。例: 0.75,1.25,1.5,2）",
    )
    args = ap.parse_args()

    metrics_path = os.path.join(args.logdir, METRICS_CSV_NAME)
    if not os.path.exists(metrics_path):
        raise SystemExit(
            f"{metrics_path} が見つかりません。\n"
            f"検知評価スイープ（python3 run_experiments.py）が出力した _logs フォルダを指定してください。"
        )
    outdir = args.outdir or os.path.join(args.logdir, "analysis")

    metrics = load_metrics(metrics_path)
    print(
        f"読み込み: {metrics_path}  行数={len(metrics)}  run数={metrics['run_id'].nunique()}  "
        f"ラウンド={metrics['round'].min()}〜{metrics['round'].max()}"
    )

    os.makedirs(outdir, exist_ok=True)
    plot_auc_by_round(metrics, os.path.join(outdir, "detection_auc_by_round.png"))
    if args.round_zs:
        zs = [float(z) for z in args.round_zs.split(",")]
        plot_auc_by_round(metrics, os.path.join(outdir, "detection_auc_by_round_selected.png"), only_zs=zs)
    plot_auc_by_round_by_detector(
        metrics, os.path.join(outdir, "detection_auc_by_round_{detector}.png")
    )
    plot_pushratio_vs_auc(metrics, os.path.join(outdir, "detection_pushratio_vs_auc.png"))
    plot_pushratio_vs_auc_by_detector(
        metrics, os.path.join(outdir, "detection_pushratio_vs_auc_{detector}.png")
    )
    pushratio_corr_table(metrics, os.path.join(outdir, "detection_pushratio_corr_table.csv"))
    plot_auc_by_z(metrics, os.path.join(outdir, "detection_auc_by_z.png"), last_k=args.last_k)
    final_auc_table(metrics, os.path.join(outdir, "detection_final_auc_table.csv"), last_k=args.last_k)
    structure_table(metrics, os.path.join(outdir, "detection_structure_table.csv"), last_k=args.last_k)
    segment_auc_table(metrics, os.path.join(outdir, "detection_segment_auc_table.csv"))
    segment_pushratio_table(metrics, os.path.join(outdir, "detection_segment_pushratio_table.csv"))

    # ラウンドごとの精度（results.csv）は，あるときだけ図と表にする。
    results_path = os.path.join(args.logdir, RESULTS_CSV_NAME)
    if os.path.exists(results_path):
        results = load_results(results_path)
        plot_accuracy_by_round(results, os.path.join(outdir, "accuracy_by_round.png"))
        accuracy_table(results, os.path.join(outdir, "accuracy_table.csv"), last_k=args.last_k)
    else:
        print(f"[精度] {results_path} が無いため，精度の図と表は作りません")

    # クライアントごとのスコア（detection_updates.csv）は，あるときだけ図と表にする。
    updates_path = os.path.join(args.logdir, UPDATES_CSV_NAME)
    if os.path.exists(updates_path):
        updates = load_updates(updates_path)
        plot_signflip_d2_score_by_round(updates, os.path.join(outdir, "signflip_d2_score_by_round.png"))
        signflip_separation_table(
            updates, metrics, os.path.join(outdir, "signflip_separation_table.csv"), last_k=args.last_k
        )
        plot_signflip_norm_ratio_by_round(updates, os.path.join(outdir, "signflip_norm_ratio_by_round.png"))
        signflip_norm_dispersion_table(
            updates, os.path.join(outdir, "signflip_norm_dispersion_table.csv"), last_k=args.last_k
        )
        signflip_norm_cv_by_segment_table(updates, os.path.join(outdir, "signflip_norm_cv_by_segment_table.csv"))
        plot_alie_d2_score_by_z(updates, os.path.join(outdir, "alie_d2_score_by_z.png"), last_k=args.last_k)
        alie_d2_score_by_z_table(updates, os.path.join(outdir, "alie_d2_score_by_z_table.csv"), last_k=args.last_k)
        alie_norm_ratio_by_z_table(updates, os.path.join(outdir, "alie_norm_ratio_by_z_table.csv"), last_k=args.last_k)
        alie_shell_table(updates, metrics, os.path.join(outdir, "alie_shell_table.csv"), last_k=args.last_k)
        plot_alie_shell_by_z(updates, metrics, os.path.join(outdir, "alie_shell_by_z.png"), last_k=args.last_k)
    else:
        print(f"[符号反転の分離] {updates_path} が無いため，図と表は作りません")


# このファイルを直接実行したときだけ，main() を呼ぶ（run_experiments.py の末尾と同じ書き方）。
if __name__ == "__main__":
    main()
