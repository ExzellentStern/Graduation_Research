"""
analyze_detection.py

sotsuron_plan2.tex 4.5 / 5 対応：検知評価スイープの後処理（集計と作図）。

【このファイル1本で完結する】
実験コード（Flower / fl_detection / detection.py / pyproject.toml）には依存しない。
必要なのは numpy / pandas / matplotlib と，実験が出力したログのフォルダだけ。
ROC-AUC などの検知性能指標は実験中に計算済み（robust_strategy.py が detection.py を
使って detection_metrics.csv に記録する）なので，ここでは読み込んで集計・作図するだけ。

入力 : <logdir>/detection_metrics.csv
       1行 =（実行, ラウンド, 検知器）。ROC-AUC / AP / recall@f / precision@f と，
       spread / push_ratio / 分布構造の診断指標を同じ行に持つ。
出力 : <outdir>/（既定は <logdir>/analysis/）
  - detection_auc_by_round.png
      検知器ごとに「AUC 対 ラウンド」を (alpha, 攻撃) 別に描く（trial 平均±標準偏差）
  - detection_pushratio_vs_auc.png
      横軸 push_ratio・縦軸 AUC の散布図（中心仮説の検証用）
  - detection_final_auc_table.csv
      最終ラウンド付近の平均 AUC 一覧（alpha × 攻撃 × 検知器）
  - detection_structure_table.csv
      最終ラウンド付近の spread / push_ratio / 構造指標と AUC の対応表

ALIE は z ごとに別の攻撃条件（alie_z0.5 など）として扱う。

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
#       2. 図1：AUC がラウンドとともにどう変わるか               plot_auc_by_round
#       3. 図2：push_ratio と AUC の関係                         plot_pushratio_vs_auc
#       4. 表1：最後の数ラウンドの平均 AUC                       final_auc_table
#       5. 表2：分布構造の診断指標と AUC の対応                  structure_table
#
#   使い方
#       python analyze_detection.py <detection_metrics.csv が入っているフォルダ>
#       例: python analyze_detection.py experiments_detection_v2/_logs
#   図と表は，指定したフォルダの中の analysis/ に出力される。
#
#   読み込むデータの形
#       表（pandas の DataFrame）。1行が「ある実行の，あるラウンドの，ある検知器の成績」。
#       54条件 × 50ラウンド × 検知器4つ = 10,800 行。
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
#   DETECTORS        : 検知器の名前（ログの detector 列に入っている値）。図や表の並び順にもなる。
#   DETECTOR_LABELS  : 図に表示するときの，検知器の見やすい名前。
#   CONDITION_KEYS   : 実験条件を表す列の名前。この7つがすべて同じ行は，同じ条件の実行とみなす。
#   STRUCTURE_COLS   : 分布構造の診断指標の列の名前。
METRICS_CSV_NAME = "detection_metrics.csv"
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
# 【load_metrics の説明】
#   detection_metrics.csv を読み込み，使いやすい形に整えて返す。
#     pd.read_csv(path)   : CSV ファイルを表（DataFrame）として読み込む。
#     df[col] = np.nan    : 列が無ければ，全部が空（NaN）の列を足す（古いログでも動くようにするため）。
#
#   回し直した条件の扱い
#     ログは追記されていくので，ある条件が途中で失敗して回し直された場合，
#     失敗した実行の行と，回し直した実行の行の両方がファイルに残っている。
#     そのまま平均を取ると，同じ条件が二重に数えられる。そこで，条件ごとに
#     「最後に書かれた実行」だけを残し，それより前の実行の行は捨てる。
#       df.groupby(CONDITION_KEYS, dropna=False)["run_id"].transform("last")
#           : 条件が同じ行をグループにし，各行に「そのグループで最後に出てくる run_id」を対応させる。
#             dropna=False は，alie_z が空の条件（ALIE 以外）もグループとして扱う指定。
#       stale = df["run_id"] != last_run : 最後の実行ではない行に True が付く。
#       df = df[~stale]                  : ~ は True と False を入れ替える。つまり，最後の実行の行だけを残す。
#
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

    last_run = df.groupby(CONDITION_KEYS, dropna=False)["run_id"].transform("last")
    stale = df["run_id"] != last_run
    if stale.any():
        dropped = sorted(df.loc[stale, "run_id"].unique())
        print(f"[load_metrics] 回し直された条件の古い実行 {len(dropped)} 件を除外: {dropped}")
        df = df[~stale]

    df["attack"] = [attack_label(t, z) for t, z in zip(df["attack_type"], df["alie_z"])]
    sort_cols = CONDITION_KEYS + ["round", "detector"]
    return df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 2) 図：AUC 対 ラウンド（trial 平均±標準偏差）
# ============================================================
# 【plot_auc_by_round の説明】
#   検知性能が，学習の進み具合（ラウンド）とともにどう変わるかを見る図。論文に載せる体裁で描いている。
#     構成 : 小さなグラフを格子状に並べる。行 = 検知器（D1〜D4），列 = 攻撃（ALIE の z ごと，符号反転）。
#     1つのグラフの中 : 横軸がラウンド，縦軸が AUC。alpha ごとに線を1本ずつ描く（色・線の種類・印を変える）。
#     線           : 3回の trial の平均。
#     薄い帯       : 平均 ± 標準偏差（trial によるばらつきの幅）。
#     点線（0.5）  : でたらめに判定したときの AUC。これより上なら見分けられていて，
#                    下なら「悪性のほうが良性らしい」と判定されている。
#
#   前半は，描くものの準備（どの攻撃・どの alpha があるかを調べ，色や線の種類を決める）。
#     attacked = metrics[metrics["attack_type"] != "none"] : 攻撃なしの行を除く（悪性がいないので AUC が無い）。
#     paper_rc = {...} : 字の大きさや目盛りの向きなど，図の見た目の設定をまとめた辞書。
#   後半の3重の for で，グラフを1つずつ描いている（検知器 -> 攻撃 -> alpha の順）。
#     cell.groupby("round")["auc"].agg(["mean", "std"])
#         : ラウンドごとに，trial をまたいだ AUC の平均と標準偏差を求める。
#     ax.plot(...)         : 線を描く。 ax.fill_between(...) : 2本の線のあいだを薄く塗る（帯）。
#     .clip(0.0, 1.0)      : 値を 0〜1 の範囲に収める（帯が図の外にはみ出さないように）。
#     fig.savefig(outpath, dpi=300) : 図をファイルに保存する。dpi は解像度（大きいほど細かい）。
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


def _auc_by_round_style(attacked: pd.DataFrame):
    """攻撃の並び・表題，alpha→(色,線種,印)の対応を作る（まとめ版・検知器ごと版で共通に使う）。

    ALIE は z の小さい順に並べ, 符号反転を右端に置く。
    alpha は色（赤・青・緑）に加えて線種とマーカーでも区別し, 白黒印刷でも読めるようにする。
    """
    alie_zs = sorted(attacked.loc[attacked["attack_type"] == "alie", "alie_z"].dropna().unique())
    attacks = [attack_label("alie", z) for z in alie_zs]
    attacks += sorted(set(attacked["attack"]) - set(attacks))
    attack_titles = {"sign_flip": "Sign flip (A1)"}
    for z in alie_zs:
        title = "ALIE (A2)" if len(alie_zs) == 1 else rf"ALIE $z={z:g}$"
        attack_titles[attack_label("alie", z)] = title
    alphas = sorted(attacked["alpha"].unique(), reverse=True)  # 100 → 0.5 → 0.1
    colors = ["tab:red", "tab:blue", "tab:green"]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    style = {
        a: (colors[i % 3], linestyles[i % 3], markers[i % 3])
        for i, a in enumerate(alphas)
    }
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


def plot_auc_by_round(metrics: pd.DataFrame, outpath: str) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[plot_auc_by_round] 攻撃ありの条件が無いためスキップ")
        return

    # 行 = 検知器, 列 = 攻撃 に分け, 1パネルあたり α の3系列だけにする.
    attacks, attack_titles, alphas, style = _auc_by_round_style(attacked)

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
# 3) 図：push_ratio 対 AUC（中心仮説の検証）
# ============================================================
# 【plot_pushratio_vs_auc の説明】
#   研究の中心仮説を確かめるための散布図。
#     横軸 : push_ratio（悪性が良性の集団からどれだけ外に押し出されているか。1 より小さいと集団の内側）。
#            対数の目盛りにしている（ax.set_xscale("log")）。値の幅が広いので，小さい側も見やすくするため。
#     縦軸 : AUC。
#     点   : 1つの点が「1つの実行の，1つのラウンド」。色が alpha，印の形が攻撃の大分類
#            （sign_flip / alie）を表す。ALIE の z の強さは印の形では区別しない
#            （push_ratio 自体が強さを表すため，凡例に出すと 15通りに膨らんで見づらくなる）。
#     構成 : 検知器ごとに1枚，計4枚のグラフを 2×2 に並べる。
#     縦の点線 : push_ratio = 1（悪性が良性の広がりのちょうど縁にいる位置）。
#   「push_ratio が 1 を下回ると AUC が崩れる」という傾向が見えるかどうかを，この図で確かめる。
#     np.isfinite(metrics["push_ratio"]) : 値が普通の数値である行だけを選ぶ（空の行を除く）。
#     ax.scatter(x, y, ...)              : 点を打つ。
#     for ax, det in zip(axes.flat, DETECTORS): : 4枚のグラフと4つの検知器を，1組ずつ対応させて回す。
def _pushratio_style(attacked: pd.DataFrame):
    """alpha→色，attack_type→印の形の対応を作る（4枚まとめ版・1枚ずつ版で共通に使う）。"""
    alphas = sorted(attacked["alpha"].unique())
    cmap = plt.get_cmap("viridis")
    color_of = {a: cmap(i / max(len(alphas) - 1, 1)) for i, a in enumerate(alphas)}
    markers = ["v", "o"]
    attack_types = sorted(attacked["attack_type"].unique())
    marker_of = {atk: markers[i % len(markers)] for i, atk in enumerate(attack_types)}
    return alphas, color_of, attack_types, marker_of


def _draw_pushratio_scatter(ax, sub: pd.DataFrame, alphas, color_of, attack_types, marker_of) -> None:
    """1つの検知器ぶんの散布図を ax に描く（4枚まとめ版・1枚ずつ版で共通）。"""
    for a in alphas:
        for atk in attack_types:
            cell = sub[(sub["alpha"] == a) & (sub["attack_type"] == atk)]
            if cell.empty:
                continue
            ax.scatter(
                cell["push_ratio"],
                cell["auc"],
                s=18,
                color=color_of[a],
                marker=marker_of[atk],
                alpha=0.6,
                edgecolors="none",
                label=f"{atk}, α={a:g}",
            )
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axvline(1.0, color="crimson", lw=0.8, ls=":")
    ax.set_ylim(0.0, 1.02)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)


def plot_pushratio_vs_auc(metrics: pd.DataFrame, outpath: str) -> None:
    attacked = metrics[(metrics["attack_type"] != "none") & np.isfinite(metrics["push_ratio"])].copy()
    if attacked.empty:
        print("[plot_pushratio_vs_auc] push_ratio が有効な攻撃条件が無いためスキップ")
        return

    # ALIE は z の強さ(0.385/0.5/0.75/1.0)ごとに別の攻撃として色・形を割り当てると
    # 凡例が 攻撃5種 × alpha3種 = 15通りに膨らんで見づらくなる。z の強さはそのまま
    # push_ratio（横軸）に表れているため，凡例では sign_flip / alie の大分類だけを区別する。
    alphas, color_of, attack_types, marker_of = _pushratio_style(attacked)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, det in zip(axes.flat, DETECTORS):
        sub = attacked[attacked["detector"] == det]
        _draw_pushratio_scatter(ax, sub, alphas, color_of, attack_types, marker_of)
        ax.set_title(DETECTOR_LABELS[det])
    for ax in axes[-1]:
        ax.set_xlabel("push_ratio = push / benign_spread  (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("malicious-detection ROC-AUC")
    axes.flat[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Detection AUC vs push_ratio  (each point = one condition × round)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[saved] {outpath}")


# 【plot_pushratio_vs_auc_by_detector の説明】
#   plot_pushratio_vs_auc と中身（点の打ち方・色・形）は同じだが，4つの検知器を
#   1枚の2x2画像にまとめず，検知器ごとに別ファイルへ1枚ずつ保存する。
#   スライドや個別の説明で1検知器だけ大きく見せたいときに使う。
#     outpath_pattern : ファイル名のひな形。中の "{detector}" が d1_norm 等に置き換わる。
#                       例 "detection_pushratio_vs_auc_{detector}.png"
def plot_pushratio_vs_auc_by_detector(metrics: pd.DataFrame, outpath_pattern: str) -> None:
    attacked = metrics[(metrics["attack_type"] != "none") & np.isfinite(metrics["push_ratio"])].copy()
    if attacked.empty:
        print("[plot_pushratio_vs_auc_by_detector] push_ratio が有効な攻撃条件が無いためスキップ")
        return

    alphas, color_of, attack_types, marker_of = _pushratio_style(attacked)

    for det in DETECTORS:
        sub = attacked[attacked["detector"] == det]
        fig, ax = plt.subplots(figsize=(6, 5))
        _draw_pushratio_scatter(ax, sub, alphas, color_of, attack_types, marker_of)
        ax.set_title(DETECTOR_LABELS[det])
        ax.set_xlabel("push_ratio = push / benign_spread  (log)")
        ax.set_ylabel("malicious-detection ROC-AUC")
        ax.legend(fontsize=8, loc="lower left")
        fig.suptitle("Detection AUC vs push_ratio  (each point = one condition × round)")
        fig.tight_layout()
        outpath = outpath_pattern.format(detector=det)
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"[saved] {outpath}")


# ============================================================
# 4) 最終ラウンド付近の平均 AUC 一覧
# ============================================================
# 【final_auc_table の説明】
#   結果を1つの数値にまとめた表。行が（alpha，攻撃），列が検知器で，マスの中が平均 AUC。
#   「最後の last_k ラウンド」（既定は5ラウンド）について，ラウンドと trial をまたいで平均している。
#   最後の数ラウンドを使うのは，学習がある程度進んだ段階での検知性能を見るため。
#     max_round = attacked["round"].max()                : 最後のラウンドの番号。
#     tail = attacked[attacked["round"] > max_round - last_k] : 最後の last_k ラウンドの行だけを取り出す
#                                                              （50 ラウンドなら 46〜50）。
#     .groupby([...])["auc"].mean() : （alpha，攻撃，検知器）の組ごとに，AUC の平均を取る。
#     .pivot(index=..., columns="detector", values="auc")
#         : 縦に長い表を，検知器が列に並ぶ形に組み替える（見やすい表にする）。
#     .round(3)                     : 小数第3位までに丸める。
#     table.to_csv(outpath)         : 表を CSV ファイルに保存する。
def final_auc_table(metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[final_auc_table] 攻撃ありの条件が無いためスキップ")
        return
    max_round = attacked["round"].max()
    tail = attacked[attacked["round"] > max_round - last_k]
    table = (
        tail.groupby(["alpha", "attack", "detector"])["auc"]
        .mean()
        .reset_index()
        .pivot(index=["alpha", "attack"], columns="detector", values="auc")
        .round(3)
    )
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 最終{last_k}ラウンドの平均 AUC ===")
    print(table.to_string())


# ============================================================
# 5) 分布構造の診断指標と AUC の対応表
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
    plot_auc_by_round_by_detector(
        metrics, os.path.join(outdir, "detection_auc_by_round_{detector}.png")
    )
    plot_pushratio_vs_auc(metrics, os.path.join(outdir, "detection_pushratio_vs_auc.png"))
    plot_pushratio_vs_auc_by_detector(
        metrics, os.path.join(outdir, "detection_pushratio_vs_auc_{detector}.png")
    )
    final_auc_table(metrics, os.path.join(outdir, "detection_final_auc_table.csv"), last_k=args.last_k)
    structure_table(metrics, os.path.join(outdir, "detection_structure_table.csv"), last_k=args.last_k)


# このファイルを直接実行したときだけ，main() を呼ぶ（run_experiments.py の末尾と同じ書き方）。
if __name__ == "__main__":
    main()
