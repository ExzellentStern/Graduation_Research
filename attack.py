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
      - Krum / Bulyan: 論文の式(2) w'_1 = w_Re - λs に合わせ、受信グローバルモデル
        （w_Re）を起点に攻撃方向へλだけ押し出した点 w'_1 を、Krumに選ばれる範囲で
        最大となるよう指数拡大＋二分探索で求める。残り c-1 クライアントは w'_1 から
        距離 epsilon（デフォルト0.01）以内にランダムに摂動させ、悪性クライアントが
        完全に同一の値になって検知されやすくなるのを避ける（論文4.1節の記述通り）。
      - FedAvg など防御機構を持たない集約手法: 外れ値を除外する仕組みが
        無いため、選択回避のためのλ探索は不要で、真の更新量を
        scale_factor 倍に拡大するだけの Scale Attack を用いる。
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


def _search_krum_lambda(
    vectors: np.ndarray,
    malicious_idx: np.ndarray,
    f: int,
    anchor: np.ndarray,
    direction: np.ndarray,
    lambda_init: float,
    n_search_steps: int,
):
    """
    crafted = anchor + λ*direction の形で悪性クライアントを密集させたとき，
    Krumに選ばれる最大のλを対数スケールの走査＋二分探索で求める。
    見つからなければ None を返す。

    単調性（λが大きいほど選ばれにくくなる、等）を仮定した片側探索ではなく，
    まず広い対数スケールで候補を走査してから最良点近傍を精緻化する。
    起点(anchor)によっては小さいλ側にしか解が無いことがあるため。
    """

    def build(lam: float) -> np.ndarray:
        crafted = vectors.copy()
        base = anchor + lam * direction
        for idx in malicious_idx:
            crafted[idx] = base
        return crafted

    def is_malicious_selected(crafted: np.ndarray) -> bool:
        scores = krum_scores(crafted, f)
        selected = int(np.argmin(scores))
        return selected in malicious_idx

    scale = max(lambda_init, 1e-6)
    candidates = [0.0] + [scale * (2.0 ** k) for k in range(-8, 9)]
    best_lambda = None
    next_failing = None
    for lam in candidates:
        if is_malicious_selected(build(lam)):
            if best_lambda is None or lam > best_lambda:
                best_lambda = lam
        elif best_lambda is None or lam > best_lambda:
            if next_failing is None or lam < next_failing:
                next_failing = lam

    if best_lambda is None:
        return None

    if next_failing is not None:
        # best_lambda（選ばれる）と next_failing（選ばれない）の間を二分探索で絞り込む
        search_lo, search_hi = best_lambda, next_failing
        for _ in range(n_search_steps):
            mid = (search_lo + search_hi) / 2.0
            if is_malicious_selected(build(mid)):
                best_lambda = mid
                search_lo = mid
            else:
                search_hi = mid

    return best_lambda


def _craft_krum_attack(
    vectors: np.ndarray,
    malicious_idx: np.ndarray,
    benign_idx: np.ndarray,
    global_vector: np.ndarray,
    f: int,
    lambda_init: float = None,
    n_search_steps: int = 20,
    epsilon: float = 0.01,
    seed: int = None,
) -> np.ndarray:
    """
    Krum（およびBulyan内部の反復Krum選択）を狙った攻撃。

    Fang et al. (2020) 式(2) w'_1 = w_Re - λs に合わせ，まず受信グローバルモデル
    （global_vector, 論文でいう w_Re）を起点として，Krumに選ばれる最大のλを探す。
    s は「honestな更新の平均が，攻撃が無ければ進んでいたはずの方向」であり，
    w'_1 はその逆方向（w_Re を挟んで honest クラスタと反対側）へ押し出される。
    「選ばれる最大のλ」は，論文のTheorem 1が与える理論的上界の代わりに，
    対数スケールの走査＋二分探索で実用的に求める（_search_krum_lambda）。

    ただし w_Re 起点は，1ラウンドでの honest クラスタの移動量が大きい場合
    （反復回数が少ない本実験の設定はこれに該当しうる）や，座標ごとの符号 s が
    実際の移動方向を粗くしか近似できない高次元の場合に，どのλでも選ばれず
    解なしとなることがある（実測でも一部ラウンドで発生することを確認済み）。
    その場合は honest クラスタの重心を起点とした探索にフォールバックし，
    それでも解が無ければ初めて攻撃を諦める（honestなまま返す）。

    さらに，論文4.1節の記述（ε=0だと悪性クライアントの更新が完全に同一になり
    サーバに検知されやすくなるため，ε=0にはしない）に合わせ，探索で求めた w'_1 は
    そのまま1クライアントに割り当て，残り c-1 クライアントには w'_1 からの距離が
    epsilon 以下になるようランダムに摂動を加えた値を割り当てる（論文のデフォルトは
    epsilon=0.01）。
    """
    benign = vectors[benign_idx]
    benign_center = benign.mean(axis=0)
    # Fangの s：honestな更新の平均が，攻撃が無ければ進んでいたはずの方向（w_Re起点）
    progress_direction = np.sign(benign_center - global_vector)
    progress_direction[progress_direction == 0] = 1.0

    # 1) Fang et al. (2020) 式(2)通り，w_Re を起点に探索する
    if lambda_init is None:
        typical_dist = float(np.linalg.norm(benign_center - global_vector))
        primary_lambda_init = max(typical_dist, 1e-6) * 2.0
    else:
        primary_lambda_init = lambda_init

    best_lambda = _search_krum_lambda(
        vectors, malicious_idx, f,
        anchor=global_vector, direction=-progress_direction,
        lambda_init=primary_lambda_init, n_search_steps=n_search_steps,
    )
    if best_lambda is not None:
        w1 = global_vector - best_lambda * progress_direction
    else:
        # 2) w_Re起点で解が無い場合は，honestクラスタ重心を起点にフォールバックする
        #    （押し出す方向は同じ「自然な進行方向の逆」のまま）
        if lambda_init is None:
            typical_spread = float(np.linalg.norm(benign - benign_center, axis=1).mean())
            fallback_lambda_init = max(typical_spread, 1e-6) * 2.0
        else:
            fallback_lambda_init = lambda_init

        fallback_lambda = _search_krum_lambda(
            vectors, malicious_idx, f,
            anchor=benign_center, direction=-progress_direction,
            lambda_init=fallback_lambda_init, n_search_steps=n_search_steps,
        )
        if fallback_lambda is None:
            # どちらの起点でもどのλでもKrumに選ばれない場合は攻撃を諦める
            return vectors.copy()
        w1 = benign_center - fallback_lambda * progress_direction

    crafted = vectors.copy()
    rng = np.random.default_rng(seed)
    for pos, idx in enumerate(malicious_idx):
        if pos == 0:
            crafted[idx] = w1
            continue
        # w1 からの距離が epsilon 以下になるよう，ランダムな方向・半径で摂動する
        direction_eps = rng.normal(size=w1.shape)
        norm = np.linalg.norm(direction_eps)
        if norm > 0:
            direction_eps = direction_eps / norm
        radius = rng.uniform(0.0, epsilon)
        crafted[idx] = w1 + radius * direction_eps
    return crafted


def _craft_scale_attack(
    vectors: np.ndarray,
    malicious_idx: np.ndarray,
    global_vector: np.ndarray,
    scale_factor: float = 10.0,
) -> np.ndarray:
    """
    FedAvg など防御機構を持たない集約手法を狙った攻撃。
    各悪性クライアントの真の更新量 delta = local - global を scale_factor 倍に
    拡大した値を採用する（client_app.py の scale_delta と同じ考え方を，
    サーバ側で vectors 全体に対して適用したもの）。単純平均は外れ値を一切
    除外しないため，Krum/Bulyan攻撃のような「選ばれる範囲でのλ探索」は不要で，
    このように大きくスケールするだけで平均を大きく歪められる。
    """
    crafted = vectors.copy()
    for idx in malicious_idx:
        delta = vectors[idx] - global_vector
        crafted[idx] = global_vector + scale_factor * delta
    return crafted


def adaptive_attack(
    vectors: np.ndarray,
    malicious_idx: Sequence[int],
    global_vector: np.ndarray,
    aggregation_method: str,
    f: int,
    krum_epsilon: float = 0.01,
    seed: int = None,
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
        krum_epsilon: Krum/Bulyan攻撃で，悪性クライアント間に許容する距離の上限
            （Fang et al. (2020) のデフォルトepsilon=0.01。0にすると悪性クライアントが
            完全に同一の値になり検知されやすくなるため使わない）。
        seed: krum_epsilonによるランダム摂動の再現性用シード。
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
        return _craft_krum_attack(
            vectors, malicious_idx, benign_idx, global_vector, f=f, epsilon=krum_epsilon, seed=seed
        )
    else:
        # fedavg など防御機構を持たない集約手法には Scale Attack を適用する
        # （外れ値を除外する仕組みが無いため，選択回避のためのλ探索は不要）
        return _craft_scale_attack(vectors, malicious_idx, global_vector)
