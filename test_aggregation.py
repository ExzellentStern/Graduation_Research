"""
test_aggregation.py

aggregation.py の単体テスト（実験計画書 7.2「小さいダミー更新で各集約の単体テストをする」対応）

torch を使わず、numpy のダミーベクトルだけで
FedAvg / Median / Trimmed Mean / Krum / Bulyan の挙動を検証する。
（state_dict <-> ベクトル変換の関数は torch 依存のため、実環境（flwr run）側で確認する）
"""

import numpy as np

from aggregation import (
    fedavg_aggregate,
    median_aggregate,
    trimmed_mean_aggregate,
    krum_aggregate,
    bulyan_aggregate,
    aggregate,
)


def make_dummy_updates(n_honest: int, n_malicious: int, dim: int = 5, seed: int = 0):
    """
    正常クライアントは 0付近の小さな乱数、悪性クライアントは明らかに外れた値（+100）を返す。
    honest の集合に近い値を返す集約手法ほど「頑健」と言える。
    """
    rng = np.random.default_rng(seed)
    honest = rng.normal(loc=0.0, scale=0.1, size=(n_honest, dim))
    malicious = np.full((n_malicious, dim), 100.0) + rng.normal(scale=0.1, size=(n_malicious, dim))
    updates = np.concatenate([honest, malicious], axis=0)
    honest_mean = honest.mean(axis=0)
    return updates, honest_mean, n_honest, n_malicious


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


def test_fedavg_baseline_is_pulled_by_outliers():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3)
    result = fedavg_aggregate(updates)  # 等重み平均
    # FedAvg は悪性クライアントの影響をそのまま受けるので honest_mean から大きくズレるはず
    dist = np.linalg.norm(result - honest_mean)
    check("FedAvg（無防備）は外れ値に引っ張られる", dist > 10.0, f"dist={dist:.3f}")


def test_median_is_robust():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3)
    result = median_aggregate(updates)
    dist = np.linalg.norm(result - honest_mean)
    check("Median は honest_mean に近い", dist < 1.0, f"dist={dist:.3f}")


def test_trimmed_mean_is_robust():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3)
    result = trimmed_mean_aggregate(updates, beta=n_m)  # 悪性数と同じだけ両端をtrim
    dist = np.linalg.norm(result - honest_mean)
    check("Trimmed Mean は honest_mean に近い", dist < 1.0, f"dist={dist:.3f}")


def test_trimmed_mean_rejects_invalid_beta():
    updates, _, n_h, n_m = make_dummy_updates(n_honest=3, n_malicious=1)
    n = n_h + n_m
    try:
        trimmed_mean_aggregate(updates, beta=n)  # 明らかに大きすぎるbeta
        ok = False
    except ValueError:
        ok = True
    check("Trimmed Mean は不正な beta を拒否する", ok)


def test_krum_selects_honest_client():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3)
    result, selected_idx = krum_aggregate(updates, f=n_m, multi_k=1)
    dist = np.linalg.norm(result - honest_mean)
    is_honest_selected = all(idx < n_h for idx in selected_idx)
    check("Krum は honest クライアントを選ぶ", is_honest_selected, f"selected={selected_idx}")
    check("Krum の出力は honest_mean に近い", dist < 1.0, f"dist={dist:.3f}")


def test_krum_raises_when_n_too_small():
    # n=2, f=1 -> 計算可能な下限 n>=f+2=3 を下回るため例外になるはず
    updates, _, n_h, n_m = make_dummy_updates(n_honest=1, n_malicious=1)
    try:
        krum_aggregate(updates, f=n_m)
        ok = False
    except ValueError:
        ok = True
    check("Krum は n < f+2 のとき例外を出す", ok)


def test_bulyan_is_robust():
    # Bulyan には n >= 4f+3 が必要。f=1 -> n>=7 なので honest=7, malicious=1 (n=8) で検証
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=1, seed=1)
    result, selected_idx = bulyan_aggregate(updates, f=n_m)
    dist = np.linalg.norm(result - honest_mean)
    is_all_honest = all(idx < n_h for idx in selected_idx)
    check("Bulyan は悪性クライアントを除外する", is_all_honest, f"selected={selected_idx}")
    check("Bulyan の出力は honest_mean に近い", dist < 1.0, f"dist={dist:.3f}")


def test_bulyan_raises_when_n_too_small():
    updates, _, n_h, n_m = make_dummy_updates(n_honest=3, n_malicious=1)
    try:
        bulyan_aggregate(updates, f=n_m)
        ok = False
    except ValueError:
        ok = True
    check("Bulyan は n < 4f+3 のとき例外を出す", ok)


def test_dispatcher_matches_direct_calls():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3, seed=2)

    for method, direct in [
        ("fedavg", fedavg_aggregate(updates)),
        ("median", median_aggregate(updates)),
        ("trimmed_mean", trimmed_mean_aggregate(updates, beta=n_m)),
    ]:
        via_dispatch = aggregate(method, updates, f=n_m)
        check(f"aggregate('{method}') が直接呼び出しと一致", np.allclose(via_dispatch, direct))

    try:
        aggregate("unknown_method", updates)
        ok = False
    except ValueError:
        ok = True
    check("aggregate() は未知の method で例外を出す", ok)


if __name__ == "__main__":
    tests = [
        test_fedavg_baseline_is_pulled_by_outliers,
        test_median_is_robust,
        test_trimmed_mean_is_robust,
        test_trimmed_mean_rejects_invalid_beta,
        test_krum_selects_honest_client,
        test_krum_raises_when_n_too_small,
        test_bulyan_is_robust,
        test_bulyan_raises_when_n_too_small,
        test_dispatcher_matches_direct_calls,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
