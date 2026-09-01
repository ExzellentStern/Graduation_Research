#!/usr/bin/env python3
"""
report/regen_ch7_assets.py

feature/windows_experiments ブランチの experiments/_logs/ から取得した
20 trial スイープ (logs_experiments_sweep/results.csv) をもとに，
卒論第7章「実験」の

  * 図 (result/<method>_<split>_by_ratio.png) 10 枚
  * ベースライン収束確認図 (result/baseline_convergence.png) 1 枚
  * 付録の Round 別 Accuracy/Loss 表 (LaTeX 片) 10 個
  * 第7章冒頭の最終 Accuracy 一覧表・収束診断表
  * 本文執筆用の統計量サマリ (report/ch7_stats.txt)

を再生成する．

このスイープは Fang et al. (2020) の設定に寄せた構成
（CNN・素朴な SGD・Fang 方式 Non-IID・batch 32・各条件 20 trial）だが，
通信ラウンド数が 20（Fang は 2,000 iterations）であり，攻撃なしの
ベースラインすら収束していない．本スクリプトはその事実を数値・図として
可視化することも目的とする．

実行:
    python3 report/regen_ch7_assets.py
"""

import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "logs_experiments_sweep", "results.csv")
FIG_DIR = os.path.join(ROOT, "result")
TEX_DIR = os.path.join(HERE, "ch7_tables")
STATS_PATH = os.path.join(HERE, "ch7_stats.txt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TEX_DIR, exist_ok=True)

TRIALS = [str(i) for i in range(1, 21)]
RATIOS = ["0.0", "0.1", "0.2", "0.3"]
METHODS = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
SPLITS = ["iid", "non_iid"]

METHOD_LABEL = {
    "fedavg": "FedAvg",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "bulyan": "Bulyan",
}
SPLIT_LABEL = {"iid": "IID", "non_iid": "Non-IID"}
SPLIT_KEY = {"iid": "iid", "non_iid": "noniid"}
RATIO_COLOR = {"0.0": "#1f77b4", "0.1": "#2ca02c", "0.2": "#ff7f0e", "0.3": "#d62728"}
METHOD_COLOR = dict(zip(METHODS, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]))

# ------------------------------------------------------------------
# load: D[(method, split, ratio)][trial][round] = (acc, loss)
# ------------------------------------------------------------------
D = defaultdict(lambda: defaultdict(dict))
with open(DATA, newline="") as f:
    for r in csv.DictReader(f):
        key = (r["aggregation_method"], r["data_split"], r["malicious_ratio"])
        D[key][r["trial"]][int(r["round"])] = (
            float(r["accuracy"]),
            float(r["loss_func"]),
        )


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def round_stats(method, split, ratio):
    """[(round, acc_mean, acc_sd, loss_mean, loss_sd, n), ...] over round 0..20."""
    per = D.get((method, split, ratio), {})
    out = []
    for rnd in range(21):
        accs, losses = [], []
        for t in TRIALS:
            if rnd in per.get(t, {}):
                a, l = per[t][rnd]
                accs.append(a)
                losses.append(l)
        if not accs:
            continue
        out.append(
            (rnd, st.mean(accs), sd(accs), st.mean(losses), sd(losses), len(accs))
        )
    return out


def has(method, split, ratio):
    return bool(D.get((method, split, ratio)))


def final_row(method, split, ratio):
    rs = {row[0]: row for row in round_stats(method, split, ratio)}
    return rs.get(20)


def end_slope(method, split, ratio, lo=16, hi=20):
    rs = {row[0]: row[1] for row in round_stats(method, split, ratio)}
    if lo in rs and hi in rs:
        return (rs[hi] - rs[lo]) / (hi - lo)
    return None


# ------------------------------------------------------------------
# per-(method, split) figures
# ------------------------------------------------------------------
def make_method_figure(method, split):
    ratios = [r for r in RATIOS if has(method, split, r)]
    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ratio in ratios:
        rs = round_stats(method, split, ratio)
        xs = [x[0] for x in rs]
        am, asd = [x[1] for x in rs], [x[2] for x in rs]
        lm, lsd = [x[3] for x in rs], [x[4] for x in rs]
        c = RATIO_COLOR[ratio]
        ax_a.plot(xs, am, marker="o", ms=3, color=c, label=f"ratio {ratio}")
        ax_a.fill_between(xs, [m - s for m, s in zip(am, asd)],
                          [m + s for m, s in zip(am, asd)], color=c, alpha=0.15)
        ax_l.plot(xs, lm, marker="o", ms=3, color=c, label=f"ratio {ratio}")
        ax_l.fill_between(xs, [m - s for m, s in zip(lm, lsd)],
                          [m + s for m, s in zip(lm, lsd)], color=c, alpha=0.15)
    ax_a.set_xlabel("Round"); ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1); ax_a.set_xlim(0, 20); ax_a.grid(True, alpha=0.3)
    ax_l.set_xlabel("Round"); ax_l.set_ylabel("Loss")
    ax_l.set_xlim(0, 20); ax_l.set_ylim(bottom=0); ax_l.grid(True, alpha=0.3)
    h, lab = ax_a.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab),
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{method}_{SPLIT_KEY[split]}_by_ratio.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(out, ROOT))


def make_baseline_figure():
    """全手法の malicious ratio 0.0 の Accuracy 推移（収束していないことの可視化）."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, split in zip(axes, SPLITS):
        for method in METHODS:
            if not has(method, split, "0.0"):
                continue
            rs = round_stats(method, split, "0.0")
            xs = [x[0] for x in rs]
            am = [x[1] for x in rs]
            ax.plot(xs, am, marker="o", ms=3, color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method])
        ax.set_title(f"{SPLIT_LABEL[split]} (malicious ratio 0.0)", fontsize=10)
        ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.set_xlim(0, 20); ax.grid(True, alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab),
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "baseline_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(out, ROOT))


# ------------------------------------------------------------------
# appendix tables (round-by-round, trial mean +/- sd)
# ------------------------------------------------------------------
def make_table(method, split):
    ratios = [r for r in RATIOS if has(method, split, r)]
    ncol = len(ratios)
    label = f"tab:{method}_{SPLIT_KEY[split]}_trialmean_by_ratio"
    cap = (
        f"{METHOD_LABEL[method]}・{SPLIT_LABEL[split]} における "
        f"malicious ratio 別の Round-Accuracy と Round-Loss"
        f"（各条件 20 trial の平均$\\pm$標準偏差）"
    )
    if method == "bulyan":
        cap += (
            "．Malicious Ratio 0.3（$f=30$）はロバスト性条件 $n \\geq 4f+3$"
            "（$100 < 123$）を満たさず，実行対象から除外している"
        )

    rs_by_ratio = {r: {row[0]: row for row in round_stats(method, split, r)} for r in ratios}

    L = []
    L.append("\\begin{table}[H]")
    L.append("\\centering")
    L.append(f"\\caption{{{cap}}}")
    L.append(f"\\label{{{label}}}")
    L.append("")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{3pt}")
    L.append("\\renewcommand{\\arraystretch}{0.90}")
    L.append("")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{c " + " ".join(["cc"] * ncol) + "}")
    L.append("")
    L.append("\\toprule")
    L.append("")
    L.append("& " + "\n& ".join(f"\\multicolumn{{2}}{{c}}{{Ratio {r}}}" for r in ratios) + " \\\\")
    L.append("")
    for i in range(ncol):
        L.append(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}")
    L.append("")
    L.append("Round")
    for _ in ratios:
        L.append("& Acc. & Loss")
    L.append(" \\\\")
    L.append("")
    L.append("\\midrule")
    L.append("")
    for rnd in range(21):
        cells = [str(rnd)]
        for r in ratios:
            row = rs_by_ratio[r].get(rnd)
            if row is None:
                cells.append("--- & ---")
                continue
            _, am, asd, lm, lsd, _n = row
            cells.append(f"{am:.3f}$\\pm${asd:.3f} & {lm:.3f}$\\pm${lsd:.3f}")
        L.append(" & ".join(cells) + " \\\\")
    L.append("")
    L.append("\\bottomrule")
    L.append("")
    L.append("\\end{tabular}")
    L.append("}")
    L.append("\\end{table}")
    L.append("")

    out = os.path.join(TEX_DIR, f"{method}_{SPLIT_KEY[split]}.tex")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("saved", os.path.relpath(out, ROOT))


# ------------------------------------------------------------------
# ch7 summary table: final accuracy (method x ratio, IID / Non-IID)
# ------------------------------------------------------------------
def make_summary_table():
    L = []
    L.append("\\begin{table}[H]")
    L.append("\\centering")
    L.append(
        "\\caption{各条件における最終ラウンド（Round 20）の Accuracy"
        "（各条件 20 trial の平均$\\pm$標準偏差）．"
        "本スイープは通信ラウンド数が不足しており（\\S\\,\\ref{subsec:ch7_caveat}），"
        "攻撃なし（Ratio 0.0）でも学習が収束していない点に注意}"
    )
    L.append("\\label{tab:ch7_final_accuracy_summary}")
    L.append("")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\renewcommand{\\arraystretch}{1.15}")
    L.append("")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{l cccc}")
    L.append("\\toprule")
    L.append("集約手法 & Ratio 0.0 & Ratio 0.1 & Ratio 0.2 & Ratio 0.3 \\\\")
    for split in SPLITS:
        L.append("\\midrule")
        L.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{SPLIT_LABEL[split]}}}}} \\\\")
        L.append("\\midrule")
        for method in METHODS:
            cells = [METHOD_LABEL[method]]
            for ratio in RATIOS:
                row = final_row(method, split, ratio)
                if row is None:
                    cells.append("---")
                else:
                    cells.append(f"{row[1]:.3f}$\\pm${row[2]:.3f}")
            L.append(" & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}%")
    L.append("}")
    L.append(
        "\\par\\vspace{2pt}\\footnotesize \\noindent "
        "Bulyan の Ratio 0.3 はロバスト性条件 $n \\geq 4f+3$ を満たさないため未実行．"
    )
    L.append("\\end{table}")
    L.append("")
    out = os.path.join(TEX_DIR, "summary_final_accuracy.tex")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("saved", os.path.relpath(out, ROOT))


# ------------------------------------------------------------------
# ch7 convergence-diagnostic table: baseline acc + end slope
# ------------------------------------------------------------------
def make_convergence_table():
    L = []
    L.append("\\begin{table}[H]")
    L.append("\\centering")
    L.append(
        "\\caption{攻撃なし（malicious ratio 0.0）条件における学習の到達度．"
        "Round 20 の Accuracy が低く，かつ Round 16--20 の Accuracy がなお"
        "毎ラウンド増加し続けていることは，通信ラウンド数 20 が学習の収束に"
        "対して不足していることを示す（各条件 20 trial 平均）}"
    )
    L.append("\\label{tab:ch7_convergence_check}")
    L.append("")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\renewcommand{\\arraystretch}{1.1}")
    L.append("")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{l cc cc}")
    L.append("\\toprule")
    L.append("& \\multicolumn{2}{c}{IID} & \\multicolumn{2}{c}{Non-IID} \\\\")
    L.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    L.append("集約手法 & Acc.@20 & $\\Delta$Acc./round & Acc.@20 & $\\Delta$Acc./round \\\\")
    L.append("\\midrule")
    for method in METHODS:
        cells = [METHOD_LABEL[method]]
        for split in SPLITS:
            row = final_row(method, split, "0.0")
            slope = end_slope(method, split, "0.0")
            if row is None:
                cells += ["---", "---"]
            else:
                cells += [f"{row[1]:.3f}", f"{slope:+.4f}"]
        L.append(" & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}%")
    L.append("}")
    L.append("\\end{table}")
    L.append("")
    out = os.path.join(TEX_DIR, "convergence_check.tex")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("saved", os.path.relpath(out, ROOT))


# ------------------------------------------------------------------
# stats dump for prose
# ------------------------------------------------------------------
def dump_stats():
    out = []
    out.append("=== FINAL ROUND (20) : trial mean +/- sample sd ===")
    for split in SPLITS:
        out.append(f"\n--- {SPLIT_LABEL[split]} ---")
        out.append(f"{'method':13s} {'ratio':5s}  {'acc':>14s}  {'loss':>14s}   slope16-20")
        for method in METHODS:
            for ratio in RATIOS:
                row = final_row(method, split, ratio)
                if row is None:
                    out.append(f"{method:13s} {ratio:5s}  (not run)")
                    continue
                _, am, asd, lm, lsd, n = row
                sl = end_slope(method, split, ratio)
                out.append(
                    f"{method:13s} {ratio:5s}  {am:.3f}+/-{asd:.3f}  {lm:.3f}+/-{lsd:.3f}"
                    f"   {sl:+.4f}  (n={n})"
                )

    out.append("\n\n=== RATIO EFFECT (acc@20 relative to ratio 0.0) ===")
    for split in SPLITS:
        out.append(f"\n--- {SPLIT_LABEL[split]} ---")
        for method in METHODS:
            base = final_row(method, split, "0.0")
            if base is None:
                continue
            deltas = []
            for ratio in RATIOS:
                row = final_row(method, split, ratio)
                if row is None:
                    deltas.append(f"r{ratio}=--")
                else:
                    deltas.append(f"r{ratio}={row[1] - base[1]:+.3f}")
            out.append(f"{method:13s}  " + "  ".join(deltas))

    out.append("\n\n=== TRAJECTORY (acc at rounds 0,5,10,15,20) ===")
    for split in SPLITS:
        out.append(f"\n--- {SPLIT_LABEL[split]} ---")
        for method in METHODS:
            for ratio in RATIOS:
                rs = {row[0]: row[1] for row in round_stats(method, split, ratio)}
                if not rs:
                    continue
                pts = "  ".join(f"r{k}:{rs.get(k, float('nan')):.3f}" for k in (0, 5, 10, 15, 20))
                out.append(f"{method:13s} {ratio:5s}  {pts}")

    text = "\n".join(out)
    with open(STATS_PATH, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print("\nwrote", os.path.relpath(STATS_PATH, ROOT))


def main():
    for method in METHODS:
        for split in SPLITS:
            if has(method, split, "0.0"):
                make_method_figure(method, split)
            make_table(method, split)
    make_baseline_figure()
    make_summary_table()
    make_convergence_table()
    dump_stats()


if __name__ == "__main__":
    main()
