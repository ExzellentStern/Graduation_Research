"""
generate_sections.py

result/results.csv から，指定した集約手法・データ分割の組ごとに
（実験条件 / 表 / 図 / 結果）を1章としてまとめる LaTeX コードを自動生成する。

result.tex 内で手作業により作成した 1章（FedAvg・IID）, 2章（FedAvg・Non-IID）
と同じ体裁を，median / trimmed_mean / krum / bulyan の各集約手法に対して
機械的に再現するためのスクリプト。数値の手書き転記によるミスを避ける目的で作成。

実行方法:
    python3 result/generate_sections.py > result/generated_sections.tex
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")

METHOD_DISPLAY = {
    "fedavg": "FedAvg",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "bulyan": "Bulyan",
}
SPLIT_DISPLAY = {"iid": "IID", "non_iid": "Non-IID"}
RATIOS = [0.0, 0.1, 0.2, 0.3]

# 対象とする（集約手法, データ分割）の組。既に手作業で作成済みの
# (fedavg, iid) / (fedavg, non_iid) は含めない。
TARGETS = [
    ("median", "iid"),
    ("median", "non_iid"),
    ("trimmed_mean", "iid"),
    ("trimmed_mean", "non_iid"),
    ("krum", "iid"),
    ("krum", "non_iid"),
    ("bulyan", "iid"),
    ("bulyan", "non_iid"),
]


def fmt(x):
    return f"{x:.4f}"


def build_table(df, method, split, ratio):
    cond = df[
        (df["aggregation_method"] == method)
        & (df["data_split"] == split)
        & (df["malicious_ratio"] == ratio)
    ]
    seeds = sorted(cond["seed"].unique())
    # 一部の Seed でログが途中欠損している場合があるため，
    # 全体の最大ラウンド（通常 20）まで行を用意し，欠損セルは "--" とする。
    overall_max_round = int(cond["round"].max())
    rounds = list(range(0, overall_max_round + 1))

    label = f"tab:{method}_{split}_ratio{ratio}"
    split_disp = SPLIT_DISPLAY[split]
    method_disp = METHOD_DISPLAY[method]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append("")
    lines.append(
        rf"\caption{{{method_disp}（{split_disp}，悪性クライアント比率 {ratio}）における"
    )
    lines.append(r"Accuracy および Loss}")
    lines.append("")
    lines.append(rf"\label{{{label}}}")
    lines.append("")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append("")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{c cc cc cc}")
    lines.append("")
    lines.append(r"\toprule")
    lines.append("")
    lines.append(
        r"& \multicolumn{2}{c}{Seed 1}"
        "\n& \\multicolumn{2}{c}{Seed 2}"
        "\n& \\multicolumn{2}{c}{Seed 3} \\\\"
    )
    lines.append("")
    lines.append(r"\cmidrule(lr){2-3}")
    lines.append(r"\cmidrule(lr){4-5}")
    lines.append(r"\cmidrule(lr){6-7}")
    lines.append("")
    lines.append("Round\n& Acc. & Loss\n& Acc. & Loss\n& Acc. & Loss \\\\")
    lines.append("")
    lines.append(r"\midrule")
    lines.append("")

    for r in rounds:
        row_cells = [str(r)]
        for s in seeds:
            cell = cond[(cond["round"] == r) & (cond["seed"] == s)]
            if cell.empty:
                row_cells.append("--")
                row_cells.append("--")
            else:
                acc = cell["accuracy"].iloc[0]
                loss = cell["loss_func"].iloc[0]
                row_cells.append(fmt(acc))
                row_cells.append(fmt(loss))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append("")
    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append("")
    lines.append(r"\end{table}")
    return "\n".join(lines), label, seeds, rounds


def build_incomplete_note(df, method, split, ratio):
    cond = df[
        (df["aggregation_method"] == method)
        & (df["data_split"] == split)
        & (df["malicious_ratio"] == ratio)
    ]
    overall_max = int(cond["round"].max())
    notes = []
    for s in sorted(cond["seed"].unique()):
        seed_max = int(cond[cond["seed"] == s]["round"].max())
        if seed_max < overall_max:
            notes.append((s, seed_max))
    if not notes:
        return ""
    parts = "，".join(f"Seed {s}（Round {m} まで）" for s, m in notes)
    return (
        f"なお，{parts} は Round {overall_max} 到達前にログが途切れており，"
        "実験が途中で終了したものと考えられる（原因不明のため，欠損箇所は表中で「--」と表記した）．"
    )


DROP_THRESHOLD = 0.02  # ピーク比でこれ以上低下していれば「不安定」として言及する


def _per_seed_peak_and_final(cond):
    """Seedごとに (ピーク accuracy, ピーク round, 最終 round, 最終 accuracy) を返す。"""
    info = []
    for seed, sg in cond.groupby("seed"):
        sg = sg.sort_values("round")
        peak_idx = sg["accuracy"].idxmax()
        peak_acc = sg.loc[peak_idx, "accuracy"]
        peak_round = int(sg.loc[peak_idx, "round"])
        final_round = int(sg["round"].max())
        final_acc = sg.loc[sg["round"] == final_round, "accuracy"].iloc[0]
        info.append((seed, peak_acc, peak_round, final_round, final_acc))
    return info


def build_stats_sentence(df, method, split, ratio):
    cond = df[
        (df["aggregation_method"] == method)
        & (df["data_split"] == split)
        & (df["malicious_ratio"] == ratio)
    ]
    r0 = cond[cond["round"] == cond["round"].min()]
    r1 = cond[cond["round"] == 1] if 1 in cond["round"].values else None
    rmax = cond[cond["round"] == cond["round"].max()]

    acc0_min, acc0_max = r0["accuracy"].min(), r0["accuracy"].max()
    accmax_min, accmax_max = rmax["accuracy"].min(), rmax["accuracy"].max()
    lossmax_min, lossmax_max = rmax["loss_func"].min(), rmax["loss_func"].max()
    final_round = int(cond["round"].max())

    sentences = []
    if r1 is not None and not r1.empty:
        acc1_min, acc1_max = r1["accuracy"].min(), r1["accuracy"].max()
        sentences.append(
            f"学習開始時（Round 0）の Accuracy は約 {fmt(acc0_min)}--{fmt(acc0_max)} であったが，"
            f"Round 1 では約 {fmt(acc1_min)}--{fmt(acc1_max)} まで上昇した．"
        )
    else:
        sentences.append(
            f"学習開始時（Round 0）の Accuracy は約 {fmt(acc0_min)}--{fmt(acc0_max)} であった．"
        )

    # ピーク後に低下しているかどうかを Seed ごとに判定する
    peak_info = _per_seed_peak_and_final(cond)
    declining = [
        (seed, peak_acc, peak_round, final_round_s, final_acc)
        for seed, peak_acc, peak_round, final_round_s, final_acc in peak_info
        if peak_acc - final_acc >= DROP_THRESHOLD
    ]

    if declining:
        peak_round_min = min(p for _, _, p, _, _ in declining)
        peak_round_max = max(p for _, _, p, _, _ in declining)
        peak_round_desc = (
            f"Round {peak_round_min}" if peak_round_min == peak_round_max
            else f"Round {peak_round_min}--{peak_round_max}"
        )
        sentences.append(
            f"その後 {peak_round_desc} 付近にかけて Accuracy は"
            f"最大約 {fmt(max(pa for _, pa, _, _, _ in declining))} まで上昇したが，"
            "その後は低下に転じ，"
            f"Round {final_round} 時点では約 {fmt(accmax_min)}--{fmt(accmax_max)} まで悪化した．"
        )
        if method == "krum":
            sentences.append(
                r"標準的な Krum（本実験では multi\_k=1）は，複数クライアントの更新を平均せず"
                "スコアが最小となる1クライアントの更新をそのまま次ラウンドの大域モデルとして採用するため，"
                "平均化によるノイズ低減効果を持たない．"
                "Adaptive attack は過去のラウンドの大域モデルを観測しながら悪性更新を作成するため，"
                "学習が進むにつれて「多数派に近い」と Krum に誤認識されやすい悪性更新を生成できるようになり，"
                "一定ラウンド以降に悪性クライアントの更新が選択される頻度が高まって"
                "Accuracy の低下を招いたと考えられる．"
            )
        else:
            sentences.append(
                "学習後半にかけて Accuracy が低下する非単調な挙動が見られており，"
                "Adaptive attack が学習の進行とともに集約結果へ影響を及ぼしている可能性が示唆される．"
            )
    else:
        sentences.append(
            f"その後も学習ラウンドの進行に伴って Accuracy は継続的に向上し，"
            f"Round {final_round} ではすべての Seed において約 {fmt(accmax_min)}--{fmt(accmax_max)} に達した．"
        )

    sentences.append(
        f"Round {final_round} 時点の Loss は約 {fmt(lossmax_min)}--{fmt(lossmax_max)} であった．"
    )

    # ratio=0.0（悪性クライアントなし）を基準とした頑健性の比較
    if ratio > 0.0:
        baseline = df[
            (df["aggregation_method"] == method)
            & (df["data_split"] == split)
            & (df["malicious_ratio"] == 0.0)
        ]
        if not baseline.empty:
            base_final_round = int(baseline["round"].max())
            base_rmax = baseline[baseline["round"] == base_final_round]
            base_mean = base_rmax["accuracy"].mean()
            cur_mean = rmax["accuracy"].mean()
            diff = base_mean - cur_mean
            if diff >= DROP_THRESHOLD:
                sentences.append(
                    f"悪性クライアント比率 0.0 の場合（最終 Accuracy 平均 約 {fmt(base_mean)}）と比較すると，"
                    f"本条件では約 {fmt(diff)} ポイント低い水準にとどまっており，"
                    f"悪性クライアント比率 {ratio} における攻撃の影響が最終的な Accuracy にも"
                    "現れていることが分かる．"
                )
            else:
                sentences.append(
                    f"悪性クライアント比率 0.0 の場合（最終 Accuracy 平均 約 {fmt(base_mean)}）と比較しても"
                    "最終的な Accuracy に大きな差は見られず，本条件下でも学習は概ね安定していたといえる．"
                )

    return " ".join(sentences)


def build_subsection(df, method, split, ratio, first_in_section):
    method_disp = METHOD_DISPLAY[method]
    split_disp = SPLIT_DISPLAY[split]

    cond = df[
        (df["aggregation_method"] == method)
        & (df["data_split"] == split)
        & (df["malicious_ratio"] == ratio)
    ]
    max_round = int(cond["round"].max()) if not cond.empty else -1

    out = []
    if not first_in_section:
        out.append(r"\clearpage")
    out.append(r"% =========================================================")
    out.append(
        rf"\subsection{{{method_disp}：{split_disp}・悪性クライアント比率 {ratio}}}"
    )
    out.append(r"% =========================================================")
    out.append("")
    out.append(r"\subsubsection{実験条件}")
    out.append("")
    out.append(r"\begin{itemize}")
    out.append(rf"    \item \textbf{{Aggregation method:}} {method_disp}")
    out.append(rf"    \item \textbf{{Data split:}} {split_disp}")
    out.append(rf"    \item \textbf{{Malicious ratio:}} {ratio}")
    out.append(r"    \item \textbf{Attack:} Adaptive attack")
    out.append(r"    \item \textbf{Number of rounds:} 20")
    out.append(r"    \item \textbf{Seeds:} 1, 2, 3")
    out.append(r"\end{itemize}")
    out.append("")

    if max_round < 20:
        # Bulyan の n >= 4f+3 制約などで学習が破綻したケース
        out.append(r"\subsubsection{結果}")
        out.append("")
        out.append(
            rf"悪性クライアント比率 {ratio}（クライアント数 100，悪性クライアント数 "
            rf"$f={int(round(ratio*100))}$）の条件では，{method_disp} の頑健性条件"
            r"（$n \geq 4f+3$）を満たさず"
            rf"（$100 < 4 \times {int(round(ratio*100))} + 3 = {4*int(round(ratio*100))+3}$），"
            r"Round 0 の初期評価以降に学習が破綻し，"
            r"Round 1 以降のログが記録されなかった．"
            rf"そのため本条件については表・図による詳細な報告を省略する．"
        )
        out.append("")
        return "\n".join(out), None

    table_tex, label, seeds, rounds = build_table(df, method, split, ratio)
    out.append(r"% =========================================================")
    out.append(r"% 実験結果表")
    out.append(r"% =========================================================")
    out.append("")
    out.append(table_tex)

    fig_name = f"seed_curves_{method}_{split}_ratio{ratio}.png"
    fig_label = f"fig:seed_curves_{method}_{split}_ratio{ratio}"
    out.append(r"\begin{figure}[H]")
    out.append(r"    \centering")
    out.append(rf"    \includegraphics[width=\columnwidth]{{{fig_name}}}")
    out.append(
        rf"    \caption{{Seed ごとの Round-Accuracy および Round-Loss の推移"
        rf"（{split_disp}，悪性クライアント比率 {ratio}）}}"
    )
    out.append(rf"    \label{{{fig_label}}}")
    out.append(r"\end{figure}")
    out.append(r"% =========================================================")
    out.append(r"\subsubsection{結果}")
    out.append(r"% =========================================================")
    out.append("")
    out.append(rf"表\ref{{{label}}}に，各ラウンドにおける")
    out.append(r"Accuracy および Loss を示す．")
    out.append("")
    out.append(build_stats_sentence(df, method, split, ratio))
    note = build_incomplete_note(df, method, split, ratio)
    if note:
        out.append("")
        out.append(note)
    out.append("")

    return "\n".join(out), fig_name


def build_section(df, method, split, is_first_section):
    method_disp = METHOD_DISPLAY[method]
    split_disp = SPLIT_DISPLAY[split]

    out = []
    if not is_first_section:
        out.append(r"\clearpage")
    out.append(r"% =========================================================")
    out.append(
        rf"\section{{{method_disp}（{split_disp}環境）における悪性クライアント比率別の評価}}"
    )
    out.append(r"% =========================================================")
    out.append("")

    fig_names = []
    for i, ratio in enumerate(RATIOS):
        sub_tex, fig_name = build_subsection(df, method, split, ratio, first_in_section=(i == 0))
        out.append(sub_tex)
        if fig_name:
            fig_names.append((method, split, ratio, fig_name))

    return "\n".join(out), fig_names


def main():
    df = pd.read_csv(RESULTS_CSV)

    all_tex = []
    all_fig_jobs = []
    for i, (method, split) in enumerate(TARGETS):
        sec_tex, fig_jobs = build_section(df, method, split, is_first_section=(i == 0))
        all_tex.append(sec_tex)
        all_fig_jobs.extend(fig_jobs)

    sys.stdout.write("\n\n".join(all_tex))
    sys.stdout.write("\n")

    # 生成が必要な図のジョブ一覧を stderr に出力（シェルスクリプト側で利用）
    for method, split, ratio, fig_name in all_fig_jobs:
        print(f"PLOTJOB\t{method}\t{split}\t{ratio}\t{fig_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
