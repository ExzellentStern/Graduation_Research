"""
test_attack.py

attack.py の単体テスト（sotsuron_plan2.tex 4.3 対応）
- sign_flip_delta: 更新量(Δ)ベースの単純攻撃（torchのダミーテンソルで検証）
- alie_attack / alie_z_max: ALIE 攻撃（numpyのダミー更新で検証）

Scale 攻撃・Adaptive 攻撃のテストは unused_features/test_attack_adaptive.py にある。
"""

import numpy as np
import torch

from fl_detection.attack import sign_flip_delta, alie_attack, alie_z_max


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


# ============================================================
# sign_flip_delta （torch dict ベース）
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


# ============================================================
# alie_attack: 準備用ヘルパー
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
# alie_attack（sotsuron_plan2.tex 4.3 A2）
# ============================================================
def test_alie_attack_payload_is_mu_minus_z_sigma():
    vectors, _, malicious_idx, honest_idx = make_dummy_round(n_honest=8, n_malicious=3, dim=6, seed=1)
    z = 1.0
    crafted = alie_attack(vectors, malicious_idx, benign_idx=honest_idx, z=z)

    benign = vectors[honest_idx]
    expected = benign.mean(axis=0) - z * benign.std(axis=0)
    for idx in malicious_idx:
        check(f"malicious[{idx}] は mu - z*sigma", np.allclose(crafted[idx], expected))
    check("悪性は全員同一ベクトル", np.allclose(crafted[malicious_idx[0]], crafted[malicious_idx[-1]]))
    check("honest分は無変更", np.allclose(crafted[honest_idx], vectors[honest_idx]))


def test_alie_attack_stays_within_benign_range():
    vectors, _, malicious_idx, honest_idx = make_dummy_round(n_honest=10, n_malicious=3, dim=8, seed=2)
    crafted = alie_attack(vectors, malicious_idx, benign_idx=honest_idx, z=1.0)
    benign = vectors[honest_idx]
    lo, hi = benign.min(axis=0), benign.max(axis=0)
    for idx in malicious_idx:
        within = np.all(crafted[idx] >= lo - 1e-9) and np.all(crafted[idx] <= hi + 1e-9)
        check(f"malicious[{idx}] はhonestの座標ごと範囲内に収まる（ステルス）", within,
              f"crafted={crafted[idx]}")


def test_alie_z_max_matches_baruch_algorithm3():
    # n=20, f=4 -> s = floor(20/2+1) - 4 = 7, z_max = Phi^{-1}(13/20)
    check("n=20, f=4 で z_max ≒ 0.385", abs(alie_z_max(20, 4) - 0.3853) < 1e-3, f"{alie_z_max(20, 4)}")
    check("悪性が増えるほど z_max は大きくなる", alie_z_max(20, 6) > alie_z_max(20, 4))
    try:
        alie_z_max(20, 11)  # s <= 0：悪性が過半数で定義できない
        check("定義できない (n, f) では ValueError", False)
    except ValueError:
        check("定義できない (n, f) では ValueError", True)


def test_alie_attack_noop_when_no_malicious():
    vectors, _, _, _ = make_dummy_round()
    all_idx = list(range(len(vectors)))
    check("悪性0人なら無変更", np.allclose(alie_attack(vectors, [], benign_idx=all_idx, z=1.0), vectors))

if __name__ == "__main__":
    tests = [
        test_sign_flip_delta,
        test_alie_attack_payload_is_mu_minus_z_sigma,
        test_alie_attack_stays_within_benign_range,
        test_alie_z_max_matches_baruch_algorithm3,
        test_alie_attack_noop_when_no_malicious,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
