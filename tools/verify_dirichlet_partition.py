"""
tools/verify_dirichlet_partition.py

research_plan_v2.tex 5 実装手順(1) 対応：
Dirichlet 分割の結果を「クライアント別のクラス数ヒストグラム」で目視確認する。

pytorchexample/task.py の _get_partitions をそのまま呼ぶので，学習時と同一の分割
（同じ trial なら同じ結果）を検証できる。

出力:
  - 標準出力にクライアント×クラスの件数表と要約統計
  - results_detection/dirichlet_partition_a{alpha}_trial{trial}.png（ヒートマップ）

使い方:
  python3 tools/verify_dirichlet_partition.py --alpha 0.1 --clients 20 --trial 1
  python3 tools/verify_dirichlet_partition.py --alpha 100 0.5 0.1 --clients 20 --trial 1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorchexample.task import _get_partitions, _load_train_dataset  # noqa: E402


def summarize_one(alpha: float, clients: int, trial: int, outdir: str) -> None:
    labels = np.asarray(_load_train_dataset()["label"])
    num_classes = int(labels.max()) + 1

    parts = _get_partitions(
        num_partitions=clients, data_split="dirichlet", alpha=alpha, p=0.5, trial=trial
    )

    counts = np.zeros((clients, num_classes), dtype=int)
    for k, idx in enumerate(parts):
        c = np.bincount(labels[idx], minlength=num_classes)
        counts[k] = c

    sizes = counts.sum(axis=1)
    classes_present = (counts > 0).sum(axis=1)

    print(f"\n================ Dirichlet alpha={alpha:g}  clients={clients}  trial={trial} ================")
    header = "client |" + "".join(f"{c:6d}" for c in range(num_classes)) + " |   total  #cls"
    print(header)
    print("-" * len(header))
    for k in range(clients):
        row = f"{k:6d} |" + "".join(f"{counts[k, c]:6d}" for c in range(num_classes))
        print(row + f" | {sizes[k]:7d} {classes_present[k]:5d}")
    print("-" * len(header))
    print(
        f"サンプル数/クライアント: min={sizes.min()}  max={sizes.max()} "
        f"mean={sizes.mean():.1f}  std={sizes.std():.1f}"
    )
    print(
        f"保有クラス数/クライアント: min={classes_present.min()} "
        f"max={classes_present.max()}  mean={classes_present.mean():.2f}  "
        f"(全{num_classes}クラス中)"
    )

    # ---- ヒートマップ ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * num_classes, 1.5 + 0.28 * clients))
        row_frac = counts / np.clip(sizes[:, None], 1, None)
        im = ax.imshow(row_frac, aspect="auto", cmap="magma")
        ax.set_xlabel("class label")
        ax.set_ylabel("client id")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(clients))
        ax.set_title(f"Dirichlet α={alpha:g}, {clients} clients, trial {trial}\n(per-client class fraction)")
        fig.colorbar(im, ax=ax, label="fraction of client's data")
        fig.tight_layout()
        path = os.path.join(outdir, f"dirichlet_partition_a{alpha:g}_trial{trial}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[saved] {path}")
    except Exception as exc:
        print(f"[warn] ヒートマップ生成に失敗しました: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dirichlet 分割の目視確認")
    ap.add_argument("--alpha", type=float, nargs="+", default=[100.0, 0.5, 0.1])
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--outdir", default="results_detection")
    args = ap.parse_args()

    for a in args.alpha:
        summarize_one(a, args.clients, args.trial, args.outdir)


if __name__ == "__main__":
    main()
