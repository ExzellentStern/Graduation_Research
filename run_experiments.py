"""
run_experiments.py

実験計画書 7.6 対応：全実験条件を自動で回すオーケストレーションスクリプト。

- aggregation / malicious_ratio / data_split / seed の一覧を定義し、全組み合わせ
  （5 x 4 x 2 x 3 = 120通り）について `flwr run` をサブプロセスとして実行する。
- 実験計画書5.2の比較条件（集約手法・悪性クライアント割合・データ分布）に沿っており、
  attack-type は本研究の主眼である "adaptive"（Adaptive Model Poisoning Attack）に固定する
  （5.2の比較条件に攻撃種別は含まれていないため）。
- 各実験は experiments/<条件名>/ に出力先を分ける（final_model.pt / fixed_eval が
  実験間で上書きされないようにする、実験計画書7.6「各実験ごとに出力ディレクトリを分ける」対応）。
- 同一条件を3回繰り返す際、seed はデータ分割と悪性クライアント選出の両方に連動させる
  （malicious-seed にも同じ値を渡す）。これにより「どのデータ分割か」「どのクライアントが
  悪性か」の両方が3回の繰り返しで変わり、より代表性のある平均が得られる。

【完了判定・再開について】
`flwr run` はサブプロセスとしては例外発生時でも終了コード0を返すことがある
（Flower内部でエラーがログに出るだけでプロセス自体は正常終了扱いになるため）。
そのため成功判定は終了コードではなく、期待される出力ファイル
（experiments/<条件名>/final_model.pt）が実際に生成されたかどうかで行う。
このファイルが既に存在する条件は「完了済み」としてスキップするため、
スクリプトを中断して再実行すれば未完了分だけ再開できる。

【失敗の記録について】
成功しなかった条件は experiments/failures.csv に記録し、スイープ全体は止めずに次へ進む。
例：Bulyanは悪性クライアント割合30%・クライアント数100だと n >= 4f+3 を満たせず、
robust_strategy.py が意図的にValueErrorを送出する（既知のBulyanの限界であり、バグではない）。

実行方法:
    python3 run_experiments.py            # 全120条件を実行（未完了分のみ）
    python3 run_experiments.py --quick     # 動作確認用の小規模設定で試す
    python3 run_experiments.py --phase1    # report/mechanism_analysis_plan.tex のPhase 1を実行
"""

import argparse
import csv
import os
import subprocess
from datetime import datetime

AGGREGATION_METHODS = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
MALICIOUS_RATIOS = [0.0, 0.1, 0.2, 0.3]
DATA_SPLITS = ["iid", "non_iid"]
SEEDS = [1, 2, 3]  # 実験計画書5.2：各条件について3回ずつ

NUM_CLIENTS = 100
NUM_SERVER_ROUNDS = 20  # 予備実験で決定した値に差し替えること（実験計画書6章）
ATTACK_TYPE = "adaptive"  # 本研究の主眼（実験計画書4.4）。5.2の比較条件に攻撃種別は含まれない

EXPERIMENTS_DIR = "experiments"
FAILURES_CSV_NAME = "failures.csv"

# ============================================================
# report/mechanism_analysis_plan.tex：Phase 1（探索）の条件定義
#
# 全条件を対象に、既存4段階のMalicious Ratioに加えて、臨界点探索のため
# 手法ごとにRatioを細分化する（Seedは全条件3のまま）。
# 出力先は experiments/ とは別の experiments_phase1/ に分ける。
# 既存 experiments/ の final_model.pt には診断ログ（logs/diagnostics.csv）が
# 含まれていない（本セッションでロギングを追加する前の実行のため）ため、
# 同じ出力先を使い回すと is_done() で誤ってスキップされてしまう。
# ============================================================
PHASE1_EXPERIMENTS_DIR = "experiments_phase1"

PHASE1_RATIOS = {
    # 手法名 -> {データ分布 -> Malicious Ratio一覧}（分布で変えない場合は "default" のみ指定）
    "fedavg": {"default": [0.0, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]},
    "median": {"default": [0.0, 0.1, 0.2, 0.3]},
    "trimmed_mean": {
        "iid": [0.0, 0.1, 0.2, 0.3],
        "non_iid": [0.0, 0.1, 0.2, 0.225, 0.25, 0.275, 0.3],
    },
    # Krumは n>=f+2 であれば計算可能だが、本研究で検討する範囲（0.0-0.3）を0.05刻みにする
    "krum": {"default": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
    # Bulyanは n>=4f+3（n=100 では f<=24 = ratio<=0.24）を満たす範囲のみ
    "bulyan": {"default": [0.0, 0.05, 0.1, 0.15, 0.2]},
}


def phase1_ratios_for(aggregation: str, data_split: str) -> list:
    table = PHASE1_RATIOS[aggregation]
    return table.get(data_split, table.get("default"))


def phase1_conditions():
    """(aggregation, data_split, ratio, seed) を Phase 1 の条件分だけ列挙する。"""
    for aggregation in AGGREGATION_METHODS:
        for data_split in DATA_SPLITS:
            for ratio in phase1_ratios_for(aggregation, data_split):
                for seed in SEEDS:
                    yield aggregation, data_split, ratio, seed


def condition_name(aggregation: str, data_split: str, ratio: float, seed: int) -> str:
    return f"{aggregation}_{data_split}_ratio{ratio}_seed{seed}"


def is_done(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "final_model.pt"))


def record_failure(name: str, cmd: list, output_tail: str, experiments_dir: str = EXPERIMENTS_DIR) -> None:
    os.makedirs(experiments_dir, exist_ok=True)
    failures_csv = os.path.join(experiments_dir, FAILURES_CSV_NAME)
    file_exists = os.path.exists(failures_csv)
    with open(failures_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "condition", "command", "output_tail"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name,
            " ".join(cmd),
            output_tail,
        ])


def run_one(
    aggregation: str,
    data_split: str,
    ratio: float,
    seed: int,
    num_clients: int,
    num_rounds: int,
    federation: str,
    experiments_dir: str = EXPERIMENTS_DIR,
) -> str:
    """1条件を実行する。戻り値は "done" / "skipped" / "failed"。"""
    name = condition_name(aggregation, data_split, ratio, seed)
    output_dir = os.path.join(experiments_dir, name)

    if is_done(output_dir):
        print(f"[SKIP] {name}（{output_dir}/final_model.pt が既に存在＝完了済み）")
        return "skipped"

    os.makedirs(output_dir, exist_ok=True)

    run_config = (
        f'num-server-rounds={num_rounds} '
        f'aggregation-method="{aggregation}" '
        f'data-split="{data_split}" '
        f'malicious-ratio={ratio} '
        f'malicious-seed={seed} '
        f'seed={seed} '
        f'attack-type="{ATTACK_TYPE}" '
        f'num-clients={num_clients} '
        f'output-dir="{output_dir}"'
    )

    cmd = [
        "flwr", "run", ".", federation,
        "--run-config", run_config,
        "--federation-config", f"num-supernodes={num_clients}",
        "--stream",  # --streamを付けないと flwr run は完了を待たずに即座に返ってしまう
    ]

    print(f"[RUN] {name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if is_done(output_dir):
        print(f"[DONE] {name}")
        return "done"

    # final_model.pt が無ければ失敗とみなす（終了コードは当てにできないため）
    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    output_tail = "\n".join(combined_output.strip().splitlines()[-30:])
    print(f"[FAILED] {name}")
    print(output_tail)
    record_failure(name, cmd, output_tail, experiments_dir=experiments_dir)
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="実験計画書7.6：全実験条件の自動実行")
    parser.add_argument("--quick", action="store_true", help="動作確認用の小規模設定で試す")
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="report/mechanism_analysis_plan.tex のPhase 1（全条件＋臨界点探索用のRatio細分化、Seed3）を実行する。"
        "出力先は experiments/ とは別の experiments_phase1/ に分ける。",
    )
    parser.add_argument("--federation", default="local-simulation", help="flwr run に渡すfederation名")
    parser.add_argument(
        "--aggregations",
        default=None,
        help="実行対象のaggregation_methodをカンマ区切りで指定（未指定時は全手法）。"
        "例：--aggregations fedavg で再実験など一部手法のみ再実行したい場合に使う。",
    )
    args = parser.parse_args()

    selected_aggregations = None
    if args.aggregations is not None:
        selected_aggregations = {a.strip() for a in args.aggregations.split(",")}

    if args.quick:
        num_clients = 10
        num_rounds = 1
        experiments_dir = EXPERIMENTS_DIR
        conditions = [
            (aggregation, data_split, ratio, seed)
            for aggregation in ["fedavg", "median"]
            for data_split in ["iid"]
            for ratio in [0.0, 0.2]
            for seed in [1]
        ]
    elif args.phase1:
        num_clients = NUM_CLIENTS
        num_rounds = NUM_SERVER_ROUNDS
        experiments_dir = PHASE1_EXPERIMENTS_DIR
        conditions = list(phase1_conditions())
    else:
        num_clients = NUM_CLIENTS
        num_rounds = NUM_SERVER_ROUNDS
        experiments_dir = EXPERIMENTS_DIR
        conditions = [
            (aggregation, data_split, ratio, seed)
            for aggregation in AGGREGATION_METHODS
            for data_split in DATA_SPLITS
            for ratio in MALICIOUS_RATIOS
            for seed in SEEDS
        ]

    if selected_aggregations is not None:
        conditions = [c for c in conditions if c[0] in selected_aggregations]

    total = len(conditions)
    print(f"=== 実験条件数: {total}（出力先: {experiments_dir}/） ===\n")

    counts = {"done": 0, "skipped": 0, "failed": 0}
    for i, (aggregation, data_split, ratio, seed) in enumerate(conditions, start=1):
        print(f"[{i}/{total}]", end=" ")
        status = run_one(
            aggregation, data_split, ratio, seed,
            num_clients, num_rounds, args.federation,
            experiments_dir=experiments_dir,
        )
        counts[status] += 1

    print("\n=== 実験終了 ===")
    print(f"完了: {counts['done']} / スキップ（完了済み）: {counts['skipped']} / "
          f"失敗: {counts['failed']} / 合計: {total}")
    if counts["failed"] > 0:
        print(f"失敗した条件は {os.path.join(experiments_dir, FAILURES_CSV_NAME)} を参照してください。")


if __name__ == "__main__":
    main()
