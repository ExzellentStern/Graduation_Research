"""
plot_seed_mean_by_ratio.py

result/results.csv から，指定した aggregation / data_split の run を
malicious_ratio ごとに抽出し，複数 Seed（デフォルト 1,2,3）の平均値を
Round-Accuracy / Round-Loss の折れ線として描画する。
Seed 間のばらつきは平均 ± 標準偏差の帯（塗り）で重ねて表示する。

実行方法:
    python3 result/plot_seed_mean_by_ratio.py --aggregation trimmed_mean --split non_iid
    python3 result/plot_seed_mean_by_ratio.py --aggregation trimmed_mean --split non_iid --seeds 1,2,3
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="malicious_ratio別のRound-Accuracy/Loss（Seed平均±標準偏差）を描画する"
    )
    parser.add_argument("--aggregation", default="trimmed_mean", help="aggregation_method")
    parser.add_argument("--split", default="non_iid", help="data_split")
    parser.add_argument("--seeds", default="1,2,3", help="平均を取るSeedのカンマ区切りリスト")
    parser.add_argument(
        "--ratios",
        default=None,
        help="描画するmalicious_ratioのカンマ区切りリスト（未指定時はデータ中の全ratio）",
    )
    parser.add_argument("--csv", default=RESULTS_CSV, help="results.csv のパス")
    parser.add_argument(
        "--out",
        default=os.path.join(SCRIPT_DIR, "seed_mean_by_ratio.png"),
        help="出力先PNGパス",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df = pd.read_csv(args.csv)
    subset = df[
        (df["aggregation_method"] == args.aggregation)
        & (df["data_split"] == args.split)
        & (df["seed"].isin(seeds))
    ]
    if subset.empty:
        raise ValueError(
            f"条件に一致するデータがありません: "
            f"aggregation={args.aggregation}, split={args.split}, seeds={seeds}"
        )

    if args.ratios is not None:
        ratios = [float(r) for r in args.ratios.split(",")]
    else:
        ratios = sorted(subset["malicious_ratio"].unique())
    colors = plt.cm.tab10.colors

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(5, 7.5))

    for i, ratio in enumerate(ratios):
        ratio_df = subset[subset["malicious_ratio"] == ratio]
        stats = (
            ratio_df.groupby("round")[["accuracy", "loss_func"]]
            .agg(["mean", "std"])
            .sort_index()
        )
        rounds = stats.index.to_numpy()
        color = colors[i % len(colors)]
        label = f"malicious ratio {ratio}"

        acc_mean = stats[("accuracy", "mean")].to_numpy()
        acc_std = stats[("accuracy", "std")].fillna(0.0).to_numpy()
        ax_acc.plot(rounds, acc_mean, marker="o", markersize=3, color=color, label=label)
        ax_acc.fill_between(
            rounds, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.2,
            linewidth=0,
        )

        loss_mean = stats[("loss_func", "mean")].to_numpy()
        loss_std = stats[("loss_func", "std")].fillna(0.0).to_numpy()
        ax_loss.plot(rounds, loss_mean, marker="o", markersize=3, color=color, label=label)
        ax_loss.fill_between(
            rounds, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.2,
            linewidth=0,
        )

    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy vs Round (mean ± std)")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss vs Round (mean ± std)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    seed_label = ",".join(str(s) for s in seeds)
    fig.suptitle(f"{args.aggregation} / {args.split} / seeds={seed_label}")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
