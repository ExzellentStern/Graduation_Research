"""
attack.py

Model Poisoning Attack の実装（実験計画書 7.4 対応）

- sign_flip_delta / scale_delta:
    「更新量ベース」の単純な攻撃。1クライアントだけで完結するローカルな攻撃で、
    torch state_dict（パラメータ名→Tensor の辞書）を直接操作する関数として実装し、
    client_app.py から呼び出す。
    Δ = local - global を計算 → 攻撃を適用 → global + Δ_malicious に戻す、という
    実験計画書7.4の一連の流れは client_app.py 側で行う（このファイルは攻撃そのものだけ担当）。

- adaptive_attack:
    Fang ら [5] の Adaptive (Local) Model Poisoning Attack。
    攻撃者はサーバで使われている集約手法を把握しており、かつ悪性クライアント同士が
    結託して他クライアントの更新も参照できる（論文でいう "full knowledge" 攻撃者）
    ことを前提とする。Flower のクライアントプロセスは互いに独立しているため、
    この前提を素直に満たせるのはサーバ側だけである。そのため adaptive_attack は
    サーバ側 (robust_strategy.py の aggregate_train) から、
    「全クライアントの更新を受け取った直後・集約する前」というタイミングで呼び出し、
    悪性クライアント分のベクトルを攻撃済みベクトルに差し替える。

    aggregation.py と同様、numpy の2次元配列 (n_clients, dim) を扱う
    torch非依存の純粋関数として実装する。

    【実装上の注意（論文との対応について）】
    本実装は Fang et al. (USENIX Security 2020) が示す「集約手法の特性を突いた
    最適化ベースの攻撃」という中核アイデアを踏襲した独自実装であり、論文本文の
    数式をそのまま再現したものではない（近似・簡略化した再実装）。
      - Trimmed Mean / Median: honestな更新の座標ごとの最大値・最小値を基準に、
        攻撃方向へ「最大/最小のさらに外側」の値を割り当てる境界値攻撃。
      - Krum / Bulyan: 悪性クライアントを一点に密集させ、Krumに選ばれる範囲で
        グローバルモデルから最大限離れるように攻撃強度λを探索する。
    卒論で原論文との厳密な数式対応を主張する場合は、原論文の該当箇所と
    本実装の差分を別途確認・記載すること。
"""

from typing import Dict, Sequence

import numpy as np

from aggregation import krum_scores

try:
    import torch
except ImportError:  # torch が無い環境でも import エラーにしない（テスト用）
    torch = None


# ============================================================
# 1) sign_flip / scale （更新量ベース、クライアント側で使用）
# ============================================================
def sign_flip_delta(delta: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Δ の符号を反転する（Sign Flipping Attack）。"""
    return {k: -v for k, v in delta.items()}


def scale_delta(delta: Dict[str, "torch.Tensor"], scale_factor: float = 10.0) -> Dict[str, "torch.Tensor"]:
    """Δ を scale_factor 倍に拡大する（Scale Attack）。"""
    return {k: scale_factor * v for k, v in delta.items()}


# ============================================================
# 2) Adaptive Model Poisoning Attack （サーバ側で使用）
# ============================================================
def _attack_direction(benign: np.ndarray, global_vector: np.ndarray) -> np.ndarray:
    """
    攻撃方向 s を決定する。
    s = sign(global - mean(honest_updates))
      -> honestな更新の平均が向かう方向と逆向きにグローバルモデルを押しやる方向。
    """
    benign_mean = benign.mean(axis=0)
    s = np.sign(global_vector - benign_mean)
    s[s == 0] = 1.0  # 方向が定まらない座標は正方向にしておく（符号0だと攻撃効果が出ないため）
    return s


def _craft_trimmed_mean_median_attack(
    vectors: np.ndarray,
    malicious_idx: np.ndarray,
    benign_idx: np.ndarray,
    global_vector: np.ndarray,
    strength: float = 3.0,
) -> np.ndarray:
    """
    Trimmed Mean / Median を狙った攻撃。
    各座標について honest 更新の最大値・最小値を求め、攻撃方向へ「最大/最小のさらに外側」
    の値を悪性クライアントへ割り当てる。悪性クライアント間で外側への押し出し具合を
    ずらす（stagger）ことで、座標ごとのtrim処理で悪性値が丸ごと除外されるのを防ぐ。
    """
    benign = vectors[benign_idx]
    s = _attack_direction(benign, global_vector)

    w_max = benign.max(axis=0)
    w_min = benign.min(axis=0)
    spread = np.maximum(w_max - w_min, 1e-12)  # 0除算回避

    n_malicious = len(malicious_idx)
    crafted = vectors.copy()
    for rank, idx in enumerate(malicious_idx):
        # rank=0 は境界ギリギリ、rank が進むほど外側へ強く押し出す
        level = strength * (rank + 1) / n_malicious
        push_up = w_max + level * spread
        push_down = w_min - level * spread
        crafted[idx] = np.where(s > 0, push_up, push_down)
    return crafted


def _craft_krum_attack(
    vectors: np.ndarray,
    malicious_idx: np.ndarray,
    benign_idx: np.ndarray,
    global_vector: np.ndarray,
    f: int,
    lambda_init: float = None,
    n_search_steps: int = 20,
) -> np.ndarray:
    """
    Krum（およびBulyan内部の反復Krum選択）を狙った攻撃。
    悪性クライアント全員に同一の crafted = honest_center + λ*s を割り当てて密集させ、
    「Krumに選ばれる最大のλ」を探索することでhonestクラスタから最大限離す。
    起点は honest クラスタの中心（benign_center）にする点が重要：
    global（前ラウンドのグローバルモデル）を起点にすると、1ラウンドの学習で
    honestクラスタ自体がglobalから離れて動いている場合に、honestクラスタから
    見て「そもそも近くない」位置になってしまい、λ=0でもKrumに選ばれなくなる。
    λが大きすぎるとhonestから離れすぎて選ばれず、小さすぎると攻撃効果が薄い。
    """
    benign = vectors[benign_idx]
    s = _attack_direction(benign, global_vector)
    benign_center = benign.mean(axis=0)

    if lambda_init is None:
        # honest更新同士の典型的な距離スケールを基準にλの探索範囲を決める
        typical_dist = float(np.linalg.norm(benign - benign_center, axis=1).mean())
        lambda_init = max(typical_dist, 1e-6) * 2.0

    def build(lam: float) -> np.ndarray:
        crafted = vectors.copy()
        base = benign_center + lam * s
        for idx in malicious_idx:
            crafted[idx] = base
        return crafted

    def is_malicious_selected(crafted: np.ndarray) -> bool:
        scores = krum_scores(crafted, f)
        selected = int(np.argmin(scores))
        return selected in malicious_idx

    # まず lambda_init で選ばれるか確認し、選ばれる限り探索範囲を倍々に広げる
    lo, hi = 0.0, lambda_init
    best_crafted = None
    for _ in range(n_search_steps):
        crafted = build(hi)
        if is_malicious_selected(crafted):
            best_crafted = crafted
            lo = hi
            hi *= 2.0
        else:
            break

    # 二分探索で「選ばれる最大のλ」に絞り込む
    search_lo, search_hi = lo, hi
    for _ in range(n_search_steps):
        mid = (search_lo + search_hi) / 2.0
        crafted = build(mid)
        if is_malicious_selected(crafted):
            best_crafted = crafted
            search_lo = mid
        else:
            search_hi = mid

    if best_crafted is None:
        # どのλでもKrumに選ばれない場合は攻撃を諦め、honestな更新のまま返す
        return vectors.copy()
    return best_crafted


def adaptive_attack(
    vectors: np.ndarray,
    malicious_idx: Sequence[int],
    global_vector: np.ndarray,
    aggregation_method: str,
    f: int,
) -> np.ndarray:
    """
    Adaptive Model Poisoning Attack のディスパッチ関数。

    vectors 全体のうち malicious_idx に該当するベクトルを、集約手法に応じて
    クラフトした悪性更新へ差し替えた新しい配列を返す（honest分は変更しない）。

    Args:
        vectors: shape (n_clients, dim)。各クライアントの（攻撃前の）学習済みモデルベクトル。
        malicious_idx: 悪性クライアントのインデックス（vectors内での位置）。
        global_vector: このラウンド開始時点のグローバルモデルベクトル（攻撃前）。
        aggregation_method: サーバが使用する集約手法（"fedavg"/"median"/"trimmed_mean"/"krum"/"bulyan"）。
        f: 防御側が想定する悪性クライアント数の上限（Krum/Bulyanのスコア計算に使用）。
    """
    malicious_idx = np.asarray(sorted(set(int(i) for i in malicious_idx)), dtype=int)
    n = vectors.shape[0]
    benign_idx = np.array([i for i in range(n) if i not in set(malicious_idx.tolist())], dtype=int)

    if len(malicious_idx) == 0 or len(benign_idx) == 0:
        return vectors.copy()

    method = aggregation_method.lower()
    if method in ("median", "trimmed_mean"):
        return _craft_trimmed_mean_median_attack(vectors, malicious_idx, benign_idx, global_vector)
    elif method in ("krum", "bulyan"):
        return _craft_krum_attack(vectors, malicious_idx, benign_idx, global_vector, f=f)
    else:
        # fedavg などrobustでない集約手法は adaptive攻撃の対象外（sign_flip/scaleで十分効く）
        return vectors.copy()
