"""
test_partition.py

task.py のデータ分割 "dirichlet_equal"（枚数は全員同じ，ラベルの比率だけを偏らせる分割）の単体テスト
- 全クライアントの枚数が samples_per_client 枚ちょうどになるか
- 同じ画像が2台以上に配られていないか
- 同じ trial なら同じ分割が再現され，trial を変えると変わるか
- alpha が大きいとデータ全体の比率に近く，小さいと特定の数字に偏るか
- 配る枚数の合計が手持ちを超えるとエラーになるか

MNIST の学習用データ（初回だけダウンロードされる）を使う。
"""

import numpy as np

from fl_detection import task
from fl_detection.task import _get_partitions, _load_train_dataset

CLIENTS = 20
SIZE = 1500


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


def _labels():
    return np.asarray(_load_train_dataset()["label"])


def _class_fractions(parts, labels):
    """クライアントごとの，各数字の割合（行 = クライアント，列 = 数字）。"""
    counts = np.array([np.bincount(labels[p], minlength=10) for p in parts])
    return counts / counts.sum(axis=1, keepdims=True)


def test_equal_size_and_no_overlap():
    for alpha in (1000.0, 5.0, 1.0):
        parts = _get_partitions(CLIENTS, "dirichlet_equal", alpha, 1, SIZE)
        sizes = [len(p) for p in parts]
        all_idx = np.concatenate(parts)
        check(f"alpha={alpha}: 全員 {SIZE} 枚", sizes == [SIZE] * CLIENTS, f"min={min(sizes)} max={max(sizes)}")
        check(f"alpha={alpha}: 同じ画像の重複なし", len(set(all_idx.tolist())) == CLIENTS * SIZE)


def test_same_trial_is_reproducible():
    a = _get_partitions(CLIENTS, "dirichlet_equal", 1.0, 1, SIZE)
    task._PARTITION_CACHE.clear()  # 控えを消して，もう一度作らせる
    b = _get_partitions(CLIENTS, "dirichlet_equal", 1.0, 1, SIZE)
    c = _get_partitions(CLIENTS, "dirichlet_equal", 1.0, 2, SIZE)
    check("同じ trial なら同じ分割", all((x == y).all() for x, y in zip(a, b)))
    check("trial を変えると分割が変わる", any((x != y).any() for x, y in zip(a, c)))


def test_alpha_controls_label_skew():
    labels = _labels()
    prior = np.bincount(labels, minlength=10) / len(labels)
    top = {}
    for alpha in (1000.0, 5.0, 1.0):
        frac = _class_fractions(_get_partitions(CLIENTS, "dirichlet_equal", alpha, 1, SIZE), labels)
        top[alpha] = float(frac.max(axis=1).mean())  # 一番多い数字が占める割合（クライアント平均）
        if alpha == 1000.0:
            dev = float(np.abs(frac - prior).max())
            check("alpha=1000: データ全体の比率に近い", dev < 0.1, f"最大のずれ={dev:.3f}")
    check(
        "alpha が小さいほど特定の数字に偏る",
        top[1000.0] < top[5.0] < top[1.0],
        f"最多の数字の割合: a=1000 {top[1000.0]:.2f} / a=5 {top[5.0]:.2f} / a=1 {top[1.0]:.2f}",
    )


def test_too_many_samples_is_error():
    try:
        _get_partitions(CLIENTS, "dirichlet_equal", 0.1, 1, 4000)  # 20 × 4000 = 80,000 > 60,000
    except ValueError as exc:
        check("配る枚数が手持ちを超えるとエラー", True, str(exc)[:30])
    else:
        check("配る枚数が手持ちを超えるとエラー", False, "エラーになりませんでした")


if __name__ == "__main__":
    test_equal_size_and_no_overlap()
    test_same_trial_is_reproducible()
    test_alpha_controls_label_skew()
    test_too_many_samples_is_error()
    print("\nすべてのテストに合格しました。")
