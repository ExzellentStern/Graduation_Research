"""
analyze_combined.py

Phase 1（experiments_phase1/_logs/、Seed1-3、全171条件）と
Phase 2（experiments_phase2/_logs/、Seed4-10、23条件のみ）を結合し、
Phase 2対象の23条件について Seed10本（Phase1のSeed1-3 + Phase2のSeed4-10）で
再集計する。experiments_phase1/analyze_phase1.py（Phase 1単体・Seed3の分析）とは
独立した出力として experiments_phase2/combined_analysis/ に保存する。

主目的：report/mechanism_analysis_plan.tex 5.2節の検証。
Phase 1（n=3）で「Seed間ばらつきが大きい」と判定した23条件が、
Seedを10に増やしても本当にばらつきが大きいままか（＝本物の不安定性か、
単にn=3での偶然の外れ値だったのか）を確認する。

実行方法:
    python experiments_phase2/analyze_combined.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PHASE1_LOGS_DIR = os.path.join(PROJECT_ROOT, "experiments_phase1", "_logs")
PHASE2_LOGS_DIR = os.path.join(SCRIPT_DIR, "_logs")
OUT_DIR = os.path.join(SCRIPT_DIR, "combined_analysis")

# Phase 2の対象23条件（experiments_phase1/analysis/seed_variance.csv の acc_std>=0.02 由来）
# run_experiments.py の PHASE2_CONDITIONS と同一。
TARGET_CONDITIONS = [
    ("bulyan", "non_iid", 0.2),
    ("fedavg", "iid", 0.1),
    ("fedavg", "iid", 0.125),
    ("fedavg", "iid", 0.15),
    ("fedavg", "iid", 0.175),
    ("fedavg", "iid", 0.2),
    ("fedavg", "iid", 0.3),
    ("fedavg", "non_iid", 0.1),
    ("fedavg", "non_iid", 0.15),
    ("fedavg", "non_iid", 0.175),
    ("fedavg", "non_iid", 0.2),
    ("fedavg", "non_iid", 0.3),
    ("krum", "iid", 0.15),
    ("krum", "iid", 0.2),
    ("krum", "iid", 0.25),
    ("krum", "iid", 0.3),
    ("krum", "non_iid", 0.0),
    ("krum", "non_iid", 0.15),
    ("krum", "non_iid", 0.2),
    ("krum", "non_iid", 0.25),
    ("krum", "non_iid", 0.3),
    ("median", "non_iid", 0.3),
    ("trimmed_mean", "non_iid", 0.3),
]

SEED_STD_THRESHOLD = 0.02  # Phase 1分析時と同じ閾値


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _is_target(row) -> bool:
    return (row["aggregation_method"], row["data_split"], row["malicious_ratio"]) in set(
        TARGET_CONDITIONS
    )


def load_combined(filename: str) -> pd.DataFrame:
    """Phase1側は対象23条件のSeed1-3行だけ、Phase2側は全行（=対象23条件のSeed4-10）を読み、
    1つのDataFrameに結合する。"""
    p1 = pd.read_csv(os.path.join(PHASE1_LOGS_DIR, filename))
    p1 = p1[p1.apply(_is_target, axis=1)].copy()
    p1["phase"] = 1

    p2 = pd.read_csv(os.path.join(PHASE2_LOGS_DIR, filename))
    p2["phase"] = 2

    combined = pd.concat([p1, p2], ignore_index=True)
    return combined


def seed_variance_before_after(summary_combined: pd.DataFrame) -> pd.DataFrame:
    """条件ごとに、Phase1単体（Seed3）とPhase1+2結合（Seed10）のacc_std/meanを比較する。"""
    rows = []
    for method, split, ratio in TARGET_CONDITIONS:
        cond = summary_combined[
            (summary_combined["aggregation_method"] == method)
            & (summary_combined["data_split"] == split)
            & (summary_combined["malicious_ratio"] == ratio)
        ]
        seed3 = cond[cond["phase"] == 1]["final_accuracy"]
        seed10 = cond["final_accuracy"]
        rows.append(
            {
                "aggregation_method": method,
                "data_split": split,
                "malicious_ratio": ratio,
                "n_seed3": len(seed3),
                "acc_mean_seed3": seed3.mean(),
                "acc_std_seed3": seed3.std(),
                "n_seed10": len(seed10),
                "acc_mean_seed10": seed10.mean(),
                "acc_std_seed10": seed10.std(),
                "still_high_variance": seed10.std() >= SEED_STD_THRESHOLD,
            }
        )
    df = pd.DataFrame(rows).sort_values("acc_std_seed10", ascending=False).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "seed_variance_before_after.csv")
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    still_high = df[df["still_high_variance"]]
    resolved = df[~df["still_high_variance"]]
    print(
        f"\n=== Seed10でも高分散のまま（acc_std >= {SEED_STD_THRESHOLD}）: "
        f"{len(still_high)}/{len(df)}件 ==="
    )
    for _, row in still_high.iterrows():
        print(
            f"  {row['aggregation_method']:<13} {row['data_split']:<8} "
            f"ratio={row['malicious_ratio']:<6} "
            f"std: {row['acc_std_seed3']:.4f}(n=3) -> {row['acc_std_seed10']:.4f}(n=10)"
        )
    print(f"\n=== Seed10では閾値未満に収まった（n=3が偶然の外れ値だった可能性）: {len(resolved)}件 ===")
    for _, row in resolved.iterrows():
        print(
            f"  {row['aggregation_method']:<13} {row['data_split']:<8} "
            f"ratio={row['malicious_ratio']:<6} "
            f"std: {row['acc_std_seed3']:.4f}(n=3) -> {row['acc_std_seed10']:.4f}(n=10)"
        )
    return df


def seed_variance_before_after_plot(df: pd.DataFrame) -> None:
    labels = [
        f"{r.aggregation_method}\n{r.data_split}\nr={r.malicious_ratio}" for r in df.itertuples()
    ]
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], df["acc_std_seed3"], width=width, label="Seed=3 (Phase1 only)")
    ax.bar([xi + width / 2 for xi in x], df["acc_std_seed10"], width=width, label="Seed=10 (Phase1+2)")
    ax.axhline(SEED_STD_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"threshold={SEED_STD_THRESHOLD}")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("final_accuracy std across seeds")
    ax.set_title("Phase 1 (n=3) vs Phase 1+2 (n=10): seed-to-seed accuracy variance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "seed_variance_before_after.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def accuracy_by_seed_scatter(summary_combined: pd.DataFrame) -> None:
    """条件ごとに、Seed1-10それぞれのfinal_accuracyを点で並べる（Phase1/Phase2を色分け）。"""
    for method, split, ratio in TARGET_CONDITIONS:
        cond = summary_combined[
            (summary_combined["aggregation_method"] == method)
            & (summary_combined["data_split"] == split)
            & (summary_combined["malicious_ratio"] == ratio)
        ].sort_values("seed")
        if cond.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = cond["phase"].map({1: "tab:blue", 2: "tab:orange"})
        ax.scatter(cond["seed"], cond["final_accuracy"], c=colors, s=60, zorder=3)
        ax.axhline(cond["final_accuracy"].mean(), color="gray", linestyle="--", linewidth=1, label="mean")

        for phase, label, color in [(1, "Phase1 (seed1-3)", "tab:blue"), (2, "Phase2 (seed4-10)", "tab:orange")]:
            ax.scatter([], [], c=color, label=label)

        ax.set_xlabel("seed")
        ax.set_ylabel("final_accuracy")
        ax.set_title(f"{method} / {split} / malicious_ratio={ratio}")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fname = f"accuracy_by_seed_{method}_{split}_ratio{ratio}.png"
        out_path = os.path.join(OUT_DIR, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved: {len(TARGET_CONDITIONS)} accuracy_by_seed_*.png files under {OUT_DIR}/")


def main():
    _ensure_out_dir()

    summary_combined = load_combined("summary.csv")
    print(f"結合後 summary: {len(summary_combined)}行（Phase1側 {sum(summary_combined['phase'] == 1)} "
          f"+ Phase2側 {sum(summary_combined['phase'] == 2)}）\n")

    combined_out = os.path.join(OUT_DIR, "combined_summary.csv")
    summary_combined.to_csv(combined_out, index=False)
    print(f"saved: {combined_out}\n")

    df = seed_variance_before_after(summary_combined)
    print()
    seed_variance_before_after_plot(df)
    accuracy_by_seed_scatter(summary_combined)

    print(f"\n全ての出力: {OUT_DIR}/")


if __name__ == "__main__":
    main()
