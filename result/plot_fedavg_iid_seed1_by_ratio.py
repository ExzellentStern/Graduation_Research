"""
plot_fedavg_iid_seed1_by_ratio.py

result/results.csv から FedAvg / IID / seed=1 の run を悪性クライアント比率
（malicious_ratio）ごとに抽出し、Round-Accuracy と Round-Loss の折れ線グラフを
比較描画する（実験結果1の1.1〜1.4に対応）。

実行方法:
    python3 result/plot_fedavg_iid_seed1_by_ratio.py
    python3 result/plot_fedavg_iid_seed1_by_ratio.py --aggregation fedavg --split iid --seed 1
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="malicious_ratio別のRound-Accuracy/Loss推移を描画する"
    )
    parser.add_argument("--aggregation", default="fedavg", help="aggregation_method")
    parser.add_argument("--split", default="iid", help="data_split")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--csv", default=RESULTS_CSV, help="results.csv のパス")
    parser.add_argument(
        "--out",
        default=os.path.join(SCRIPT_DIR, "fedavg_iid_seed1_by_ratio.png"),
        help="出力先PNGパス",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    subset = df[
        (df["aggregation_method"] == args.aggregation)
        & (df["data_split"] == args.split)
        & (df["seed"] == args.seed)
    ]
    if subset.empty:
        raise ValueError(
            f"条件に一致するデータがありません: "
            f"aggregation={args.aggregation}, split={args.split}, seed={args.seed}"
        )

    ratios = sorted(subset["malicious_ratio"].unique())
    colors = plt.cm.tab10.colors

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(5, 7.5))

    for i, ratio in enumerate(ratios):
        ratio_df = subset[subset["malicious_ratio"] == ratio].sort_values("round")
        color = colors[i % len(colors)]
        ax_acc.plot(
            ratio_df["round"], ratio_df["accuracy"], marker="o", markersize=3,
            color=color, label=f"malicious ratio {ratio}",
        )
        ax_loss.plot(
            ratio_df["round"], ratio_df["loss_func"], marker="o", markersize=3,
            color=color, label=f"malicious ratio {ratio}",
        )

    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy vs Round")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss vs Round")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    fig.suptitle(f"{args.aggregation} / {args.split} / seed={args.seed}")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
