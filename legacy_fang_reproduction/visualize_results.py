"""
visualize_results.py

実験計画書 7.7 対応：logs/results.csv・logs/summary.csv から比較図を生成する。

run_experiments.py（7.6）で複数条件を実行した後、このスクリプトを実行すると
以下6種類の図を logs/plots/ に保存する。

    1. accuracy_transition.png    : ラウンドごとのAccuracy推移（悪性割合×データ分布でパネル分割）
    2. loss_transition.png        : 同上、Loss版
    3. final_accuracy_bar.png     : 最終Accuracyの棒グラフ（集約手法×悪性割合、データ分布でパネル分割）
    4. final_loss_bar.png         : 同上、Loss版
    5. iid_vs_noniid.png          : IID/Non-IID比較図（悪性割合ごとにパネル分割）
    6. aggregation_robustness.png : 集約手法ごとの比較図
                                     （悪性割合に対する最終Accuracyの頑健性曲線）

各図は複数回の繰り返し実験（seed違い）の平均値を実線・±標準偏差を陰影/誤差棒で表示する。
欠損条件（例：Bulyanがn>=4f+3を満たせず失敗した組み合わせ）はそのまま欠けて描画される。

実行方法:
    python3 visualize_results.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = "logs"
RESULTS_CSV = os.path.join(LOG_DIR, "results.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary.csv")
PLOT_DIR = os.path.join(LOG_DIR, "plots")

# 集約手法の表示順序・色を固定し、図をまたいで一貫させる
AGGREGATION_ORDER = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
COLORS = dict(zip(AGGREGATION_ORDER, plt.cm.tab10.colors[: len(AGGREGATION_ORDER)]))


def _ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def _load():
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(
            f"{RESULTS_CSV} / {SUMMARY_CSV} が見つかりません。"
            " 先に run_experiments.py（実験計画書7.6）を実行してください。"
        )
    return pd.read_csv(RESULTS_CSV), pd.read_csv(SUMMARY_CSV)


def _ordered(values, preferred_order):
    """preferred_order に含まれる値を優先順位で並べ、含まれない値は末尾に足す。"""
    ordered = [v for v in preferred_order if v in values]
    extra = sorted(v for v in values if v not in preferred_order)
    return ordered + extra


def _save(fig, out_name):
    _ensure_dir()
    path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {path}")


# ============================================================
# 1, 2) Accuracy / Loss 推移（実験計画書7.7）
# ============================================================
def plot_transition(results: pd.DataFrame, metric: str, ylabel: str, out_name: str):
    ratios = sorted(results["malicious_ratio"].unique())
    splits = _ordered(results["data_split"].unique(), ["iid", "non_iid"])
    aggregations = _ordered(results["aggregation_method"].unique(), AGGREGATION_ORDER)

    fig, axes = plt.subplots(
        len(ratios), len(splits),
        figsize=(5.5 * len(splits), 3.2 * len(ratios)),
        squeeze=False,
    )

    for i, ratio in enumerate(ratios):
        for j, split in enumerate(splits):
            ax = axes[i][j]
            subset = results[(results["malicious_ratio"] == ratio) & (results["data_split"] == split)]
            for agg in aggregations:
                agg_df = subset[subset["aggregation_method"] == agg]
                if agg_df.empty:
                    continue
                grouped = agg_df.groupby("round")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(grouped["round"], grouped["mean"], label=agg, color=COLORS.get(agg), marker="o", markersize=3)
                std = grouped["std"].fillna(0)
                ax.fill_between(
                    grouped["round"], grouped["mean"] - std, grouped["mean"] + std,
                    color=COLORS.get(agg), alpha=0.15,
                )
            ax.set_title(f"{split} / malicious={ratio:.0%}", fontsize=10)
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(aggregations), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, out_name)


# ============================================================
# 3, 4) 最終 Accuracy / Loss の棒グラフ（実験計画書7.7）
# ============================================================
def plot_final_bar(summary: pd.DataFrame, metric: str, ylabel: str, out_name: str):
    splits = _ordered(summary["data_split"].unique(), ["iid", "non_iid"])
    aggregations = _ordered(summary["aggregation_method"].unique(), AGGREGATION_ORDER)
    ratios = sorted(summary["malicious_ratio"].unique())

    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.5), squeeze=False)
    axes = axes[0]

    bar_width = 0.8 / max(len(ratios), 1)
    x = list(range(len(aggregations)))

    for j, split in enumerate(splits):
        ax = axes[j]
        subset = summary[summary["data_split"] == split]
        for k, ratio in enumerate(ratios):
            means, stds = [], []
            for agg in aggregations:
                cell = subset[(subset["aggregation_method"] == agg) & (subset["malicious_ratio"] == ratio)][metric]
                means.append(cell.mean() if not cell.empty else float("nan"))
                stds.append(cell.std() if len(cell) > 1 else 0.0)
            offsets = [xi + (k - (len(ratios) - 1) / 2) * bar_width for xi in x]
            ax.bar(offsets, means, width=bar_width, yerr=stds, label=f"malicious={ratio:.0%}", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(aggregations, rotation=20)
        ax.set_title(split)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ratios), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, out_name)


# ============================================================
# 5) IID / Non-IID 比較図（実験計画書7.7）
# ============================================================
def plot_iid_vs_noniid(summary: pd.DataFrame, out_name: str = "iid_vs_noniid.png"):
    ratios = sorted(summary["malicious_ratio"].unique())
    aggregations = _ordered(summary["aggregation_method"].unique(), AGGREGATION_ORDER)
    splits = _ordered(summary["data_split"].unique(), ["iid", "non_iid"])

    ncols = min(len(ratios), 2)
    nrows = -(-len(ratios) // ncols)  # 切り上げ
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    bar_width = 0.8 / max(len(splits), 1)
    x = list(range(len(aggregations)))

    for idx, ratio in enumerate(ratios):
        ax = axes_flat[idx]
        subset = summary[summary["malicious_ratio"] == ratio]
        for k, split in enumerate(splits):
            means, stds = [], []
            for agg in aggregations:
                cell = subset[
                    (subset["aggregation_method"] == agg) & (subset["data_split"] == split)
                ]["final_accuracy"]
                means.append(cell.mean() if not cell.empty else float("nan"))
                stds.append(cell.std() if len(cell) > 1 else 0.0)
            offsets = [xi + (k - (len(splits) - 1) / 2) * bar_width for xi in x]
            ax.bar(offsets, means, width=bar_width, yerr=stds, label=split, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(aggregations, rotation=20)
        ax.set_title(f"malicious={ratio:.0%}")
        ax.set_ylabel("Final Accuracy")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[len(ratios):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(splits), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("IID vs Non-IID: Final Accuracy", y=1.08)
    fig.tight_layout()
    _save(fig, out_name)


# ============================================================
# 6) 集約手法ごとの比較図（頑健性曲線）（実験計画書7.7）
# ============================================================
def plot_aggregation_robustness(summary: pd.DataFrame, out_name: str = "aggregation_robustness.png"):
    splits = _ordered(summary["data_split"].unique(), ["iid", "non_iid"])
    aggregations = _ordered(summary["aggregation_method"].unique(), AGGREGATION_ORDER)

    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.5), squeeze=False)
    axes = axes[0]

    for j, split in enumerate(splits):
        ax = axes[j]
        subset = summary[summary["data_split"] == split]
        for agg in aggregations:
            agg_df = subset[subset["aggregation_method"] == agg]
            if agg_df.empty:
                continue
            grouped = agg_df.groupby("malicious_ratio")["final_accuracy"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                grouped["malicious_ratio"], grouped["mean"], yerr=grouped["std"].fillna(0),
                marker="o", label=agg, color=COLORS.get(agg), capsize=3,
            )
        ax.set_title(split)
        ax.set_xlabel("Malicious Client Ratio")
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(aggregations), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Aggregation Method Robustness under Adaptive Model Poisoning Attack", y=1.1)
    fig.tight_layout()
    _save(fig, out_name)


def main():
    results, summary = _load()
    plot_transition(results, "accuracy", "Accuracy", "accuracy_transition.png")
    plot_transition(results, "loss_func", "Loss", "loss_transition.png")
    plot_final_bar(summary, "final_accuracy", "Final Accuracy", "final_accuracy_bar.png")
    plot_final_bar(summary, "final_loss", "Final Loss", "final_loss_bar.png")
    plot_iid_vs_noniid(summary)
    plot_aggregation_robustness(summary)
    print(f"\nすべての図を {PLOT_DIR} に保存しました。")


if __name__ == "__main__":
    main()
