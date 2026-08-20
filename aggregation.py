"""
aggregation.py

Robust Aggregation 手法の実装（実験計画書 7.2 対応）
- モデル重み(state_dict) <-> 1次元ベクトルの相互変換
- FedAvg（重み付き平均）を自作関数として切り出し
- Median / Trimmed Mean / Krum / Bulyan の実装
- aggregation type（文字列）で切り替え可能な aggregate() ディスパッチ関数

【設計方針】
- 集約の中核ロジック（fedavg/median/trimmed_mean/krum/bulyan）は
  numpy の2次元配列 (n_clients, dim) を入力とする「純粋関数」として実装する。
  → PyTorch に依存しないため、テストが軽量になり、将来的に別フレームワークでも流用できる。
- Robust Aggregation（Median/Trimmed Mean/Krum/Bulyan）では、
  クライアントのデータ数 nk による重み付けは行わない（全クライアント等重み）。
  ※ 悪性クライアントが nk を偽って過大申告すると重み付き平均で耐性が崩れるため、
    Byzantine-robust な文脈では等重みで扱うのが一般的（Fangら等の先行研究に準拠）。
- FedAvg のみ、元の実装通りデータ数による重み付けをオプションで行える。

【Krum / Bulyan のクライアント数の制約】
- Krum:    n >= f + 3  が必要（n - f - 2 >= 1）
- Bulyan:  n >= 4f + 3 が必要（Krumをtheta=n-2f回繰り返した上でTrimmed Meanをかけるため）
  f は「防御側が想定する悪性クライアント数の上限」であり、
  実際の malicious_ids（攻撃側の真の情報）とは独立したパラメータとして扱う。
"""

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # このモジュール単体でのテスト時など、torch が無い環境でも import エラーにしない
    torch = None


# ============================================================
# 1) state_dict <-> ベクトル変換
# ============================================================
def state_dict_to_vector(state_dict) -> Tuple[np.ndarray, List[Tuple[str, tuple]]]:
    """
    torch の state_dict（OrderedDict[str, Tensor]）を1次元 numpy ベクトルに変換する。

    Returns:
        vector: 全パラメータを連結した1次元 float64 配列
        shapes: [(key, shape), ...] 復元用のメタ情報（元の順序を保持）
    """
    shapes = []
    flat_list = []
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy()
        shapes.append((k, tuple(arr.shape)))
        flat_list.append(arr.reshape(-1).astype(np.float64))
    vector = np.concatenate(flat_list) if flat_list else np.array([], dtype=np.float64)
    return vector, shapes


def vector_to_state_dict(vector: np.ndarray, shapes: List[Tuple[str, tuple]], reference_state_dict):
    """
    1次元 numpy ベクトルを state_dict（OrderedDict[str, Tensor]）に復元する。

    Args:
        vector: state_dict_to_vector で得たベクトルと同じ並び・次元数のベクトル
        shapes: state_dict_to_vector が返した shapes
        reference_state_dict: 各パラメータの dtype を復元するための参照 state_dict
            （通常は集約対象クライアントのうちどれか1つの state_dict でよい）
    """
    if torch is None:
        raise ImportError("vector_to_state_dict には torch が必要です。")

    new_state = OrderedDict()
    cursor = 0
    for key, shape in shapes:
        numel = int(np.prod(shape)) if len(shape) > 0 else 1
        chunk = vector[cursor: cursor + numel].reshape(shape)
        cursor += numel
        ref_dtype = reference_state_dict[key].dtype
        new_state[key] = torch.tensor(chunk, dtype=ref_dtype)

    if cursor != len(vector):
        raise ValueError(
            f"ベクトル長が shapes と一致しません（消費={cursor}, 実際={len(vector)}）。"
            " state_dict の構造が全クライアントで一致しているか確認してください。"
        )
    return new_state


def stack_updates(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """複数クライアントのベクトルを (n_clients, dim) の2次元配列にまとめる。"""
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"全クライアントのベクトル長が一致していません: {lengths}")
    return np.stack(vectors, axis=0)


# ============================================================
# 2) FedAvg（重み付き平均） -- 自作関数として切り出し
# ============================================================
def fedavg_aggregate(updates: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    updates: shape (n_clients, dim)
    weights: shape (n_clients,)。None の場合は等重み平均。
             通常は各クライアントのデータ数 nk を渡す（元の FedAvg と同じ挙動）。
    """
    n = updates.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n:
            raise ValueError("weights の長さが updates のクライアント数と一致しません。")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights の合計が0以下です。")
        w = w / total
    return np.average(updates, axis=0, weights=w)


# ============================================================
# 3) Median
# ============================================================
def median_aggregate(updates: np.ndarray) -> np.ndarray:
    """各パラメータ（座標）ごとに中央値を取る。"""
    return np.median(updates, axis=0)


# ============================================================
# 4) Trimmed Mean
# ============================================================
def trimmed_mean_aggregate(updates: np.ndarray, beta: int) -> np.ndarray:
    """
    各パラメータについて、大きい方から beta 個・小さい方から beta 個を除外してから平均する。

    Args:
        beta: 片側あたりの除外数（通常は防御側が想定する悪性クライアント数 f と同じ値を使う）
    """
    n = updates.shape[0]
    if beta < 0:
        raise ValueError("beta は 0 以上である必要があります。")
    if 2 * beta >= n:
        raise ValueError(f"2*beta({2*beta}) が n({n}) 以上です。beta を小さくしてください。")
    sorted_updates = np.sort(updates, axis=0)
    if beta == 0:
        trimmed = sorted_updates
    else:
        trimmed = sorted_updates[beta: n - beta]
    return trimmed.mean(axis=0)


# ============================================================
# 5) Krum
# ============================================================
def _pairwise_sq_distances(updates: np.ndarray) -> np.ndarray:
    """全クライアント間のユークリッド距離の2乗を (n, n) 行列で返す。"""
    sq_norms = np.sum(updates ** 2, axis=1)
    dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (updates @ updates.T)
    dists = np.maximum(dists, 0.0)  # 数値誤差による微小な負値を防ぐ
    np.fill_diagonal(dists, 0.0)
    return dists


def krum_scores(updates: np.ndarray, f: int) -> np.ndarray:
    """
    各クライアント i について、自分に最も近い (n - f - 2) 個のクライアントとの
    距離2乗の合計を Krum スコアとして返す。スコアが小さいほど「他の多数派に近い」＝信頼できる。

    計算可能なための最小条件は n >= f + 2（k = n-f-2 >= 0）。
    ※ Byzantine耐性の理論的保証としてよく引用されるのは n >= 2f+3 だが、
      それより弱い「計算が破綻しない」ための下限が n >= f+2 である。
      k=0（n = f+2）の場合は近傍0個＝スコアが全クライアント0で並ぶ退化ケースとなる
      （Bulyan内部で候補が絞り込まれた終盤の反復でこの境界に達することがあるため、
        例外にはせず計算を継続できるようにしている）。
    """
    n = updates.shape[0]
    k = n - f - 2
    if k < 0:
        raise ValueError(f"Krum には n >= f + 2 が必要です（n={n}, f={f}）。")

    dists = _pairwise_sq_distances(updates)
    scores = np.zeros(n)
    for i in range(n):
        d_sorted = np.sort(dists[i])  # d_sorted[0] は自分自身との距離(=0)
        scores[i] = d_sorted[1: k + 1].sum()
    return scores


def krum_aggregate(updates: np.ndarray, f: int, multi_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    標準的な Krum（multi_k=1）は、最もスコアの低い1クライアントの更新をそのまま採用する
    （平均は取らない）。multi_k>1 を指定すると Multi-Krum（上位 multi_k 個の平均）になる。

    Returns:
        aggregated: 集約結果ベクトル
        selected_idx: 選択されたクライアントのインデックス（デバッグ・ログ用）
    """
    scores = krum_scores(updates, f)
    selected_idx = np.argsort(scores)[:multi_k]
    aggregated = updates[selected_idx].mean(axis=0)
    return aggregated, selected_idx


# ============================================================
# 6) Bulyan（Krumによる候補選択 + Trimmed Mean）
# ============================================================
def bulyan_aggregate(updates: np.ndarray, f: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Krum を繰り返し適用し、theta = n - 2f 個のクライアントを選択する
       （選択するたびにそのクライアントを候補集合から除外し、再度Krumスコアを計算する）
    2. 選択された theta 個の更新に対して、各座標で上下 f 個を除いた Trimmed Mean を取る

    Bulyan には n >= 4f + 3 が必要（各ステップで Krum が成立する条件を満たすため）。
    """
    n = updates.shape[0]

    # f=0（悪性クライアントを想定しない）場合、理論上 Bulyan は単純平均に一致する。
    # このとき theta=n となり、終盤の反復で残り候補数が1まで減って Krum が計算不能
    # （n_remaining - f - 2 < 0）になる退化ケースに当たるため、先に単純平均へ委譲する。
    if f == 0:
        return fedavg_aggregate(updates), np.arange(n)

    theta = n - 2 * f
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan には n >= 4f+3 が必要です（n={n}, f={f}）。")

    remaining_updates = updates.copy()
    remaining_global_idx = np.arange(n)
    selected_global_idx = []

    for _ in range(theta):
        scores = krum_scores(remaining_updates, f)
        best_local = int(np.argmin(scores))
        selected_global_idx.append(int(remaining_global_idx[best_local]))

        mask = np.ones(len(remaining_global_idx), dtype=bool)
        mask[best_local] = False
        remaining_updates = remaining_updates[mask]
        remaining_global_idx = remaining_global_idx[mask]

    selection = updates[selected_global_idx]  # shape (theta, dim)
    aggregated = trimmed_mean_aggregate(selection, beta=f)
    return aggregated, np.asarray(selected_global_idx)


# ============================================================
# 7) ディスパッチ関数（aggregation type で切り替え）
# ============================================================
SUPPORTED_METHODS = ("fedavg", "median", "trimmed_mean", "krum", "bulyan")


def is_bulyan_feasible(n: int, f: int) -> bool:
    """
    Bulyan が計算可能かどうか（n >= 4f+3、f=0は常にTrue）を事前に判定する。

    【設計メモ】
    悪性クライアント割合30%（f=30, n=100）の設定では 100 < 4*30+3=123 となり、
    Bulyan は理論上の前提条件を満たさない。これは実装上の不具合ではなく、
    Bulyanアルゴリズム自体の既知の限界（頑健性を保証できるクライアント数に上限がある）
    である。本研究ではこの failure を隠さず、bulyan_aggregate() が ValueError を送出する
    ようにしてある。run_experiments.py 側ではこの関数で事前判定するか、
    ValueError を捕捉して「Bulyan不成立」として結果に記録し、考察（Bulyanの限界）の
    材料として扱う方針とする。
    """
    if f == 0:
        return True
    return n >= 4 * f + 3


def aggregate(
    method: str,
    updates: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    f: int = 0,
    trimmed_mean_beta: Optional[int] = None,
    krum_multi_k: int = 1,
) -> np.ndarray:
    """
    aggregation type（文字列）に応じて集約手法を切り替えるディスパッチ関数。

    Args:
        method: "fedavg" | "median" | "trimmed_mean" | "krum" | "bulyan"
        updates: shape (n_clients, dim) の各クライアント更新
        weights: fedavg でのみ使用（データ数による重み付け等）
        f: 防御側が想定する悪性クライアント数の上限（median以外のrobust手法で使用）
        trimmed_mean_beta: 指定があれば trimmed_mean の beta として f より優先して使う
        krum_multi_k: krum で平均に使う候補数（1なら標準Krum）
    """
    name = method.lower()
    if name == "fedavg":
        return fedavg_aggregate(updates, weights)
    elif name == "median":
        return median_aggregate(updates)
    elif name == "trimmed_mean":
        beta = trimmed_mean_beta if trimmed_mean_beta is not None else f
        return trimmed_mean_aggregate(updates, beta)
    elif name == "krum":
        aggregated, _ = krum_aggregate(updates, f, multi_k=krum_multi_k)
        return aggregated
    elif name == "bulyan":
        aggregated, _ = bulyan_aggregate(updates, f)
        return aggregated
    else:
        raise ValueError(f"未対応の aggregation method です: {method!r}（対応: {SUPPORTED_METHODS}）")
