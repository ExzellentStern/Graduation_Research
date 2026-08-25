"""
plot_summary_comparison.py

result/summary.csv（各 run の最終ラウンド結果）から，集約手法を横断的に比較する図を作る。
result.tex の「総合考察」セクションに埋め込むための2枚を出力する。

    1. summary_final_accuracy_bar.png  : 集約手法 x 悪性クライアント比率 の最終Accuracy棒グラフ
                                          （IID / Non-IID でパネル分割）
    2. summary_robustness_curve.png    : 悪性クライアント比率に対する最終Accuracyの推移曲線
                                          （集約手法別，IID / Non-IID でパネル分割）

実行方法:
    python3 result/plot_summary_comparison.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "summary.csv")

AGGREGATION_ORDER = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
METHOD_DISPLAY = {
    "fedavg": "FedAvg",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "bulyan": "Bulyan",
}
SPLIT_DISPLAY = {"iid": "IID", "non_iid": "Non-IID"}
COLORS = dict(zip(AGGREGATION_ORDER, plt.cm.tab10.colors[: len(AGGREGATION_ORDER)]))


def plot_final_accuracy_bar(df, out_path):
    splits = ["iid", "non_iid"]
    ratios = sorted(df["malicious_ratio"].unique())
    aggregations = [a for a in AGGREGATION_ORDER if a in df["aggregation_method"].unique()]

    fig, axes = plt.subplots(len(splits), 1, figsize=(5.5, 4.0 * len(splits)), squeeze=False)
    axes = axes[:, 0]

    bar_width = 0.8 / max(len(ratios), 1)
    x = list(range(len(aggregations)))

    for j, split in enumerate(splits):
        ax = axes[j]
        subset = df[df["data_split"] == split]
        for k, ratio in enumerate(ratios):
            means, stds = [], []
            for agg in aggregations:
                cell = subset[
                    (subset["aggregation_method"] == agg) & (subset["malicious_ratio"] == ratio)
                ]["final_accuracy"]
                means.append(cell.mean() if not cell.empty else float("nan"))
                stds.append(cell.std() if len(cell) > 1 else 0.0)
            offsets = [xi + (k - (len(ratios) - 1) / 2) * bar_width for xi in x]
            ax.bar(offsets, means, width=bar_width, yerr=stds, label=f"malicious={ratio:.0%}", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[a] for a in aggregations], rotation=20)
        ax.set_title(SPLIT_DISPLAY[split])
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ratios), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Final Accuracy by Aggregation Method and Malicious Ratio", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_robustness_curve(df, out_path):
    splits = ["iid", "non_iid"]
    aggregations = [a for a in AGGREGATION_ORDER if a in df["aggregation_method"].unique()]

    fig, axes = plt.subplots(len(splits), 1, figsize=(5.5, 4.0 * len(splits)), squeeze=False)
    axes = axes[:, 0]

    for j, split in enumerate(splits):
        ax = axes[j]
        subset = df[df["data_split"] == split]
        for agg in aggregations:
            agg_df = subset[subset["aggregation_method"] == agg]
            if agg_df.empty:
                continue
            grouped = agg_df.groupby("malicious_ratio")["final_accuracy"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                grouped["malicious_ratio"], grouped["mean"], yerr=grouped["std"].fillna(0),
                marker="o", label=METHOD_DISPLAY[agg], color=COLORS.get(agg), capsize=3,
            )
        ax.set_title(SPLIT_DISPLAY[split])
        ax.set_xlabel("Malicious Client Ratio")
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(aggregations) // 2 + 1, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Robustness Curve: Final Accuracy vs Malicious Client Ratio", y=1.10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    df = pd.read_csv(SUMMARY_CSV)
    plot_final_accuracy_bar(df, os.path.join(SCRIPT_DIR, "summary_final_accuracy_bar.png"))
    plot_robustness_curve(df, os.path.join(SCRIPT_DIR, "summary_robustness_curve.png"))


if __name__ == "__main__":
    main()
