"""
plot_seed_curves.py

result/results.csv から、指定条件（デフォルトは fedavg / iid / malicious_ratio=0.0）の
run を抽出し、seed ごとに Round-Accuracy と Round-Loss の折れ線グラフを描画する。

実行方法:
    python3 result/plot_seed_curves.py
    python3 result/plot_seed_curves.py --aggregation fedavg --split iid --ratio 0.0
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="seed別のRound-Accuracy/Loss推移を描画する")
    parser.add_argument("--aggregation", default="fedavg", help="aggregation_method")
    parser.add_argument("--split", default="iid", help="data_split")
    parser.add_argument("--ratio", type=float, default=0.0, help="malicious_ratio")
    parser.add_argument("--csv", default=RESULTS_CSV, help="results.csv のパス")
    parser.add_argument(
        "--out", default=os.path.join(SCRIPT_DIR, "seed_curves.png"), help="出力先PNGパス"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    subset = df[
        (df["aggregation_method"] == args.aggregation)
        & (df["data_split"] == args.split)
        & (df["malicious_ratio"] == args.ratio)
    ]
    if subset.empty:
        raise ValueError(
            f"条件に一致するデータがありません: "
            f"aggregation={args.aggregation}, split={args.split}, ratio={args.ratio}"
        )

    seeds = sorted(subset["seed"].unique())
    colors = plt.cm.tab10.colors

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(5, 7.5))

    for i, seed in enumerate(seeds):
        seed_df = subset[subset["seed"] == seed].sort_values("round")
        color = colors[i % len(colors)]
        ax_acc.plot(seed_df["round"], seed_df["accuracy"], marker="o", markersize=3,
                    color=color, label=f"seed {seed}")
        ax_loss.plot(seed_df["round"], seed_df["loss_func"], marker="o", markersize=3,
                     color=color, label=f"seed {seed}")

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

    fig.suptitle(
        f"{args.aggregation} / {args.split} / malicious_ratio={args.ratio}"
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
