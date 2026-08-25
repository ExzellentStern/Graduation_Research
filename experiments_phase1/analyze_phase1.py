"""
analyze_phase1.py

report/mechanism_analysis_plan.tex の Phase 1 結果（experiments_phase1/_logs/）を分析する。
「6 今後の流れ」の 3〜5 に対応：

  3. Seed間で最終Accuracyのばらつきが大きいRatio点を特定する（Phase 2候補の抽出）
  4. push_ratio をIID / Non-IID・手法別に比較する図を作成する
  5. Krum/Bulyanの選択結果（frac_selected_malicious）とMalicious Ratio 0.0での
     精度変動を突き合わせる（benign_spread と accuracy の推移を並べて見る）

出力は experiments_phase1/analysis/ 以下に保存する。

実行方法:
    python experiments_phase1/analyze_phase1.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "_logs")
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis")

SUMMARY_CSV = os.path.join(LOGS_DIR, "summary.csv")
RESULTS_CSV = os.path.join(LOGS_DIR, "results.csv")
DIAGNOSTICS_CSV = os.path.join(LOGS_DIR, "diagnostics.csv")

METHODS = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
SPLITS = ["iid", "non_iid"]

# Phase 2候補と判定するSeed間標準偏差の閾値。
# 計画書2節「他のMalicious Ratioでは Seed間差は0.01未満」を基準に、
# それより明確に大きい0.02を境界とする。
SEED_STD_THRESHOLD = 0.02


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 3. Seed間ばらつきの特定（Phase 2候補点の抽出）
# ============================================================
def seed_variance_table(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["aggregation_method", "data_split", "malicious_ratio"])[
            "final_accuracy"
        ]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_seeds"})
        .reset_index()
    )
    grouped["acc_std"] = grouped["acc_std"].fillna(0.0)
    grouped = grouped.sort_values("acc_std", ascending=False).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "seed_variance.csv")
    grouped.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    candidates = grouped[grouped["acc_std"] >= SEED_STD_THRESHOLD]
    print(
        f"\n=== Phase 2候補点（Seed間 acc_std >= {SEED_STD_THRESHOLD}）: "
        f"{len(candidates)}件 ==="
    )
    if candidates.empty:
        print("  該当なし")
    else:
        for _, row in candidates.iterrows():
            print(
                f"  {row['aggregation_method']:<13} {row['data_split']:<8} "
                f"ratio={row['malicious_ratio']:<6} "
                f"acc_std={row['acc_std']:.4f} (n_seeds={int(row['n_seeds'])}, "
                f"range={row['min']:.3f}-{row['max']:.3f})"
            )
    return grouped


# ============================================================
# 4. push_ratio のIID / Non-IID・手法別比較
# ============================================================
def push_ratio_comparison_plot(diagnostics: pd.DataFrame) -> None:
    df = diagnostics.dropna(subset=["push_ratio"])
    df = df[df["malicious_ratio"] > 0]
    if df.empty:
        print("push_ratio: 悪性クライアントが存在する行がないためスキップ")
        return

    stats = (
        df.groupby(["aggregation_method", "data_split"])["push_ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )

    methods = [m for m in METHODS if m in stats["aggregation_method"].unique()]
    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, split in enumerate(SPLITS):
        split_stats = stats[stats["data_split"] == split].set_index("aggregation_method")
        means = [split_stats["mean"].get(m, float("nan")) for m in methods]
        stds = [split_stats["std"].get(m, 0.0) for m in methods]
        offset = (i - 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=split,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("push_ratio (mean ± std)")
    ax.set_title("push_ratio: IID vs Non-IID by aggregation method\n(mean over all rounds/seeds, malicious_ratio > 0)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="data_split")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "push_ratio_iid_vs_noniid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ============================================================
# 5. Krum/BulyanのMalicious Ratio 0.0での精度変動とbenign_spreadの突き合わせ
# ============================================================
def krum_bulyan_spread_vs_accuracy_plot(
    results: pd.DataFrame, diagnostics: pd.DataFrame
) -> None:
    for method in ["krum", "bulyan"]:
        for split in SPLITS:
            acc_df = results[
                (results["aggregation_method"] == method)
                & (results["data_split"] == split)
                & (results["malicious_ratio"] == 0.0)
            ]
            spread_df = diagnostics[
                (diagnostics["aggregation_method"] == method)
                & (diagnostics["data_split"] == split)
                & (diagnostics["malicious_ratio"] == 0.0)
            ]
            if acc_df.empty or spread_df.empty:
                print(f"{method}/{split}/ratio0.0: データなし、スキップ")
                continue

            fig, (ax_acc, ax_spread) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

            for seed, seed_df in acc_df.groupby("seed"):
                seed_df = seed_df.sort_values("round")
                ax_acc.plot(
                    seed_df["round"], seed_df["accuracy"], marker="o", markersize=3,
                    label=f"seed {seed}",
                )
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title(f"{method} / {split} / malicious_ratio=0.0")
            ax_acc.grid(True, alpha=0.3)
            ax_acc.legend(fontsize=8)

            for seed, seed_df in spread_df.groupby("seed"):
                seed_df = seed_df.sort_values("round")
                ax_spread.plot(
                    seed_df["round"], seed_df["benign_spread"], marker="o", markersize=3,
                    label=f"seed {seed}",
                )
            ax_spread.set_xlabel("Round")
            ax_spread.set_ylabel("benign_spread")
            ax_spread.grid(True, alpha=0.3)

            fig.tight_layout()
            out_path = os.path.join(OUT_DIR, f"{method}_{split}_ratio0.0_spread_vs_accuracy.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved: {out_path}")


# ============================================================
# おまけ：headline figure（Accuracy vs Malicious Ratio, 手法別・Seed平均±標準偏差）
# ============================================================
def accuracy_vs_ratio_overview_plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12, 5), sharey=True)

    for ax, split in zip(axes, SPLITS):
        split_df = summary[summary["data_split"] == split]
        for method in METHODS:
            method_df = split_df[split_df["aggregation_method"] == method]
            if method_df.empty:
                continue
            stats = (
                method_df.groupby("malicious_ratio")["final_accuracy"]
                .agg(["mean", "std"])
                .sort_index()
            )
            ax.errorbar(
                stats.index, stats["mean"], yerr=stats["std"].fillna(0.0),
                marker="o", markersize=4, capsize=3, label=method,
            )
        ax.set_xlabel("malicious_ratio")
        ax.set_title(split)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("final_accuracy (mean ± std over seeds)")
    axes[-1].legend(title="aggregation_method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Phase 1: Final Accuracy vs Malicious Ratio")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "accuracy_vs_ratio_overview.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    _ensure_out_dir()

    summary = pd.read_csv(SUMMARY_CSV)
    results = pd.read_csv(RESULTS_CSV)
    diagnostics = pd.read_csv(DIAGNOSTICS_CSV)

    print(f"summary: {len(summary)}行 / results: {len(results)}行 / diagnostics: {len(diagnostics)}行\n")

    seed_variance_table(summary)
    print()
    accuracy_vs_ratio_overview_plot(summary)
    push_ratio_comparison_plot(diagnostics)
    krum_bulyan_spread_vs_accuracy_plot(results, diagnostics)

    print(f"\n全ての出力: {OUT_DIR}/")


if __name__ == "__main__":
    main()
