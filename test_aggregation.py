"""
test_aggregation.py

aggregation.py の単体テスト（実験計画書 7.2「小さいダミー更新で各集約の単体テストをする」対応）

torch を使わず、numpy のダミーベクトルだけで FedAvg の挙動を検証する。
（state_dict <-> ベクトル変換の関数は torch 依存のため、実環境（flwr run）側で確認する）

Median / Trimmed Mean / Krum / Bulyan のテストは unused_features/test_aggregation_robust.py にある。
"""

import numpy as np

from fl_detection.aggregation import fedavg_aggregate, aggregate


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


def test_fedavg_weights_follow_data_size():
    updates = np.array([[0.0, 0.0], [10.0, 20.0]])
    result = fedavg_aggregate(updates, weights=np.array([3.0, 1.0]))
    check("FedAvg はデータ数で重み付けする", np.allclose(result, [2.5, 5.0]), f"result={result}")


def test_dispatcher_matches_direct_calls():
    updates, honest_mean, n_h, n_m = make_dummy_updates(n_honest=7, n_malicious=3, seed=2)

    via_dispatch = aggregate("fedavg", updates)
    check("aggregate('fedavg') が直接呼び出しと一致", np.allclose(via_dispatch, fedavg_aggregate(updates)))

    for method in ["median", "unknown_method"]:
        try:
            aggregate(method, updates)
            ok = False
        except ValueError:
            ok = True
        check(f"aggregate() は未対応の method（{method}）で例外を出す", ok)


if __name__ == "__main__":
    tests = [
        test_fedavg_baseline_is_pulled_by_outliers,
        test_fedavg_weights_follow_data_size,
        test_dispatcher_matches_direct_calls,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
