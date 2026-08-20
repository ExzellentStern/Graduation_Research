"""
test_attack.py

attack.py の単体テスト（実験計画書 7.4 対応）
- sign_flip_delta / scale_delta: 更新量(Δ)ベースの単純攻撃（torchのダミーテンソルで検証）
- adaptive_attack: Fangらの Adaptive Model Poisoning Attack
    （numpyのダミー更新で、各集約手法向けのクラフトが意図通りかを検証）
"""

import numpy as np
import torch

from attack import sign_flip_delta, scale_delta, adaptive_attack
from aggregation import krum_scores, median_aggregate, trimmed_mean_aggregate


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


# ============================================================
# sign_flip_delta / scale_delta （torch dict ベース）
# ============================================================
def test_sign_flip_delta():
    delta = {"w": torch.tensor([1.0, -2.0, 3.0]), "b": torch.tensor([0.5])}
    flipped = sign_flip_delta(delta)
    check(
        "sign_flip_delta は符号を反転する",
        torch.allclose(flipped["w"], torch.tensor([-1.0, 2.0, -3.0]))
        and torch.allclose(flipped["b"], torch.tensor([-0.5])),
    )
    # 元のdeltaは破壊しない
    check("元のdeltaは変更されない", torch.allclose(delta["w"], torch.tensor([1.0, -2.0, 3.0])))


def test_scale_delta():
    delta = {"w": torch.tensor([1.0, -2.0, 3.0])}
    scaled = scale_delta(delta, scale_factor=10.0)
    check(
        "scale_delta は scale_factor 倍する",
        torch.allclose(scaled["w"], torch.tensor([10.0, -20.0, 30.0])),
    )


# ============================================================
# adaptive_attack: 準備用ヘルパー
# ============================================================
def make_dummy_round(n_honest=7, n_malicious=3, dim=5, seed=0):
    """
    実際のFLに近い設定：honestな各クライアントの更新は、globalモデルから
    「小さく」動いた値になる（1ラウンドでの変化量は通常小さいため）。
    悪性クライアントも「攻撃前」は同じスケールで小さく動いた値からスタートする。
    """
    rng = np.random.default_rng(seed)
    global_vector = rng.normal(loc=0.0, scale=0.05, size=dim)
    honest = global_vector + rng.normal(loc=0.05, scale=0.02, size=(n_honest, dim))
    malicious_before_attack = global_vector + rng.normal(loc=0.05, scale=0.02, size=(n_malicious, dim))
    vectors = np.concatenate([honest, malicious_before_attack], axis=0)
    malicious_idx = list(range(n_honest, n_honest + n_malicious))
    honest_idx = list(range(n_honest))
    return vectors, global_vector, malicious_idx, honest_idx


# ============================================================
# adaptive_attack: median / trimmed_mean 向け
# ============================================================
def test_adaptive_attack_trimmed_mean_pushes_beyond_honest_range():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round()
    crafted = adaptive_attack(
        vectors, malicious_idx, global_vector, aggregation_method="trimmed_mean", f=len(malicious_idx)
    )

    honest = vectors[honest_idx]
    w_max = honest.max(axis=0)
    w_min = honest.min(axis=0)

    # honestは正方向(1.0付近)へ進んだので、攻撃方向は「global - honest_mean < 0」寄り
    # -> 悪性値は w_min よりさらに下（あるいはw_maxより上）に押し出されているはず
    for idx in malicious_idx:
        beyond_max = np.all(crafted[idx] >= w_max - 1e-9)
        beyond_min = np.all(crafted[idx] <= w_min + 1e-9)
        check(
            f"malicious[{idx}] はhonestの範囲外に押し出されている",
            beyond_max or beyond_min,
            f"crafted={crafted[idx]}, w_max={w_max}, w_min={w_min}",
        )

    # honest分は変更されていない
    check("honestな更新は変更されない", np.allclose(crafted[honest_idx], vectors[honest_idx]))


def test_adaptive_attack_trimmed_mean_shifts_result_away_from_honest():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round(n_honest=7, n_malicious=3)
    crafted = adaptive_attack(
        vectors, malicious_idx, global_vector, aggregation_method="trimmed_mean", f=len(malicious_idx)
    )

    honest_only_mean = trimmed_mean_aggregate(vectors[honest_idx], beta=0)  # honestだけの平均(参考値)
    attacked_result = trimmed_mean_aggregate(crafted, beta=len(malicious_idx))

    dist = np.linalg.norm(attacked_result - honest_only_mean)
    check(
        "adaptive攻撃はhonestのみの結果からズレを生む",
        dist > 0.01,
        f"dist={dist:.4f}",
    )


def test_adaptive_attack_median_also_crafts():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round()
    crafted = adaptive_attack(
        vectors, malicious_idx, global_vector, aggregation_method="median", f=len(malicious_idx)
    )
    check("median向けにも悪性値がcraftされる", not np.allclose(crafted[malicious_idx], vectors[malicious_idx]))


# ============================================================
# adaptive_attack: krum / bulyan 向け
# ============================================================
def test_adaptive_attack_krum_gets_selected():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round(n_honest=7, n_malicious=3)
    f = len(malicious_idx)
    crafted = adaptive_attack(vectors, malicious_idx, global_vector, aggregation_method="krum", f=f)

    scores = krum_scores(crafted, f)
    selected = int(np.argmin(scores))
    check(
        "krum向けのcraft後、悪性クライアントがKrumに選ばれる",
        selected in malicious_idx,
        f"selected={selected}, malicious_idx={malicious_idx}",
    )


def test_adaptive_attack_krum_moves_in_attack_direction():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round(n_honest=7, n_malicious=3)
    f = len(malicious_idx)
    crafted = adaptive_attack(vectors, malicious_idx, global_vector, aggregation_method="krum", f=f)

    honest = vectors[honest_idx]
    benign_center = honest.mean(axis=0)
    s = np.sign(global_vector - benign_center)  # 期待される攻撃方向

    moved = crafted[malicious_idx[0]] - benign_center
    # honestクラスタ中心から見て、攻撃方向 s と同じ向きに動いているはず（内積が正）
    check(
        "krum攻撃はhonestクラスタ中心から攻撃方向へcraftする",
        float(np.dot(moved, s)) > 0,
        f"moved={moved}, s={s}",
    )


def test_adaptive_attack_bulyan_uses_krum_style():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round(n_honest=7, n_malicious=1)
    f = len(malicious_idx)
    crafted_bulyan = adaptive_attack(vectors, malicious_idx, global_vector, aggregation_method="bulyan", f=f)
    crafted_krum = adaptive_attack(vectors, malicious_idx, global_vector, aggregation_method="krum", f=f)
    check(
        "bulyanはkrumと同じcraftingロジックを使う",
        np.allclose(crafted_bulyan, crafted_krum),
    )


# ============================================================
# adaptive_attack: 対象外の集約手法・境界ケース
# ============================================================
def test_adaptive_attack_noop_for_fedavg():
    vectors, global_vector, malicious_idx, honest_idx = make_dummy_round()
    crafted = adaptive_attack(vectors, malicious_idx, global_vector, aggregation_method="fedavg", f=len(malicious_idx))
    check("fedavgはadaptive攻撃の対象外（無変更）", np.allclose(crafted, vectors))


def test_adaptive_attack_noop_when_no_malicious():
    vectors, global_vector, _, honest_idx = make_dummy_round()
    crafted = adaptive_attack(vectors, [], global_vector, aggregation_method="trimmed_mean", f=0)
    check("悪性クライアントが0人なら無変更", np.allclose(crafted, vectors))


if __name__ == "__main__":
    tests = [
        test_sign_flip_delta,
        test_scale_delta,
        test_adaptive_attack_trimmed_mean_pushes_beyond_honest_range,
        test_adaptive_attack_trimmed_mean_shifts_result_away_from_honest,
        test_adaptive_attack_median_also_crafts,
        test_adaptive_attack_krum_gets_selected,
        test_adaptive_attack_krum_moves_in_attack_direction,
        test_adaptive_attack_bulyan_uses_krum_style,
        test_adaptive_attack_noop_for_fedavg,
        test_adaptive_attack_noop_when_no_malicious,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
