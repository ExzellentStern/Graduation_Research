"""
analyze_detection.py

research_plan_v2.tex 4.5 / 5 対応：検知評価スイープの後処理。

入力 : logs/detection_updates.csv（robust_strategy.py が1クライアント1行で追記）
出力 :
  - logs/detection_metrics.csv
      条件×ラウンド×検知器ごとの ROC-AUC / AP / recall@f / precision@f
  - results_detection/detection_auc_by_round.png
      検知器ごとに「AUC 対 ラウンド」を (alpha, 攻撃) 別に描く（trial 平均±標準偏差）
  - results_detection/detection_pushratio_vs_auc.png
      横軸 push_ratio・縦軸 AUC の散布図（中心仮説の検証用）
  - results_detection/detection_final_auc_table.csv
      最終ラウンド付近の平均 AUC 一覧（alpha × 攻撃 × 検知器）

使い方:
  python3 analyze_detection.py
  python3 analyze_detection.py --updates logs/detection_updates.csv --outdir result
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detection import (
    average_precision,
    precision_at_k,
    recall_at_k,
    roc_auc,
)

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
    "malicious_ratio",
    "trial",
]


# ============================================================
# 1) 条件×ラウンド×検知器ごとの指標を計算
# ============================================================
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = CONDITION_KEYS + ["round"]
    for keys, g in df.groupby(group_cols, dropna=False):
        cond = dict(zip(group_cols, keys))
        y = g["is_malicious"].to_numpy().astype(int)
        f = int(y.sum())  # 想定悪性数 = 実際の悪性数（recall@f / precision@f の k）
        base = dict(cond)
        base["n_clients"] = len(g)
        base["n_malicious"] = f
        base["benign_spread"] = float(np.nanmean(g["benign_spread"].to_numpy(dtype=float)))
        pr = g["push_ratio"].to_numpy(dtype=float)
        base["push_ratio"] = float(np.nanmean(pr)) if np.isfinite(pr).any() else np.nan

        for det in DETECTORS:
            s = g[det].to_numpy(dtype=float)
            rec = dict(base)
            rec["detector"] = det
            rec["auc"] = roc_auc(y, s)
            rec["ap"] = average_precision(y, s)
            rec["recall_at_f"] = recall_at_k(y, s, f)
            rec["precision_at_f"] = precision_at_k(y, s, f)
            records.append(rec)
    out = pd.DataFrame.from_records(records)
    sort_cols = [c for c in CONDITION_KEYS if c in out.columns] + ["round", "detector"]
    return out.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 2) 図：AUC 対 ラウンド（trial 平均±標準偏差）
# ============================================================
def plot_auc_by_round(metrics: pd.DataFrame, outpath: str) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[plot_auc_by_round] 攻撃ありの条件が無いためスキップ")
        return

    combos = (
        attacked[["alpha", "attack_type"]]
        .drop_duplicates()
        .sort_values(["attack_type", "alpha"])
        .to_records(index=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, det in zip(axes.flat, DETECTORS):
        sub = attacked[attacked["detector"] == det]
        for alpha, attack in combos:
            cell = sub[(sub["alpha"] == alpha) & (sub["attack_type"] == attack)]
            if cell.empty:
                continue
            agg = cell.groupby("round")["auc"].agg(["mean", "std"]).reset_index()
            label = f"{attack}, α={alpha:g}"
            ax.plot(agg["round"], agg["mean"], marker="o", ms=3, label=label)
            ax.fill_between(
                agg["round"],
                agg["mean"] - agg["std"].fillna(0.0),
                agg["mean"] + agg["std"].fillna(0.0),
                alpha=0.15,
            )
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_title(DETECTOR_LABELS[det])
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("communication round")
    for ax in axes[:, 0]:
        ax.set_ylabel("malicious-detection ROC-AUC")
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Detection AUC vs round (mean ± SD over trials)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[saved] {outpath}")


# ============================================================
# 3) 図：push_ratio 対 AUC（中心仮説の検証）
# ============================================================
def plot_pushratio_vs_auc(metrics: pd.DataFrame, outpath: str) -> None:
    attacked = metrics[(metrics["attack_type"] != "none") & np.isfinite(metrics["push_ratio"])].copy()
    if attacked.empty:
        print("[plot_pushratio_vs_auc] push_ratio が有効な攻撃条件が無いためスキップ")
        return

    alphas = sorted(attacked["alpha"].unique())
    cmap = plt.get_cmap("viridis")
    color_of = {a: cmap(i / max(len(alphas) - 1, 1)) for i, a in enumerate(alphas)}
    markers = ["o", "s", "^", "D", "v", "P"]
    attacks = sorted(attacked["attack_type"].unique())
    marker_of = {atk: markers[i % len(markers)] for i, atk in enumerate(attacks)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, det in zip(axes.flat, DETECTORS):
        sub = attacked[attacked["detector"] == det]
        for a in alphas:
            for atk in attacks:
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
        ax.set_title(DETECTOR_LABELS[det])
        ax.set_ylim(0.0, 1.02)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
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


# ============================================================
# 4) 最終ラウンド付近の平均 AUC 一覧
# ============================================================
def final_auc_table(metrics: pd.DataFrame, outpath: str, last_k: int = 5) -> None:
    attacked = metrics[metrics["attack_type"] != "none"].copy()
    if attacked.empty:
        print("[final_auc_table] 攻撃ありの条件が無いためスキップ")
        return
    max_round = attacked["round"].max()
    tail = attacked[attacked["round"] > max_round - last_k]
    table = (
        tail.groupby(["alpha", "attack_type", "detector"])["auc"]
        .mean()
        .reset_index()
        .pivot(index=["alpha", "attack_type"], columns="detector", values="auc")
        .round(3)
    )
    table.to_csv(outpath)
    print(f"[saved] {outpath}")
    print(f"\n=== 最終{last_k}ラウンドの平均 AUC ===")
    print(table.to_string())


# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="検知評価スイープの後処理")
    ap.add_argument("--updates", default=os.path.join("logs", "detection_updates.csv"))
    ap.add_argument("--metrics-out", default=os.path.join("logs", "detection_metrics.csv"))
    ap.add_argument("--outdir", default="results_detection")
    ap.add_argument("--last-k", type=int, default=5, help="最終平均を取るラウンド数")
    args = ap.parse_args()

    if not os.path.exists(args.updates):
        raise SystemExit(
            f"{args.updates} が見つかりません。先に検知評価スイープを回してください：\n"
            f"    python3 run_experiments.py --detection"
        )

    df = pd.read_csv(args.updates)
    print(f"読み込み: {args.updates}  行数={len(df)}  run数={df['run_id'].nunique()}")

    metrics = compute_metrics(df)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    metrics.to_csv(args.metrics_out, index=False)
    print(f"[saved] {args.metrics_out}  行数={len(metrics)}")

    os.makedirs(args.outdir, exist_ok=True)
    plot_auc_by_round(metrics, os.path.join(args.outdir, "detection_auc_by_round.png"))
    plot_pushratio_vs_auc(metrics, os.path.join(args.outdir, "detection_pushratio_vs_auc.png"))
    final_auc_table(metrics, os.path.join(args.outdir, "detection_final_auc_table.csv"), last_k=args.last_k)


if __name__ == "__main__":
    main()
