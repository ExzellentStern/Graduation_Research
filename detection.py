"""
detection.py

研究計画書（改訂版, sotsuron_plan2.tex）4.2--4.5 対応：
更新ベクトルレベルの異常検知（悪性更新の二値分類）と，その検知性能指標。

【設計方針】
- aggregation.py / attack.py と同じく，numpy の2次元配列 (n_clients, dim) を
  入力とする「純粋関数」として実装する（PyTorch / scikit-learn 非依存）。
  → テストが軽量になり，Flower を起動しなくても単体で検証できる。
- 入力は「更新ベクトル」 g_i = w_i - w_global（＝各クライアントのモデル更新から
  グローバルモデルを引いた差分）を想定する。robust_strategy.py 側で
  updates - global_vector を渡すこと。

【検知器】（いずれも「スコアが大きいほど悪性らしい」向きに揃える）
  D1 norm     : ノルム偏差            s_i = | ||g_i|| - median_j ||g_j|| |
  D2 cosine   : 参照方向へのコサイン距離 s_i = 1 - cos(g_i, r),  r = coord-wise median
  D3 kmeans   : 標準化した特徴量 [norm, D2, PC1, PC2] への 2-means。
                大きい方のクラスタ（＝正常とみなす）重心までの距離。
  D4 spectral : 中心化した更新行列の第1右特異ベクトル v への射影の2乗 (g_i . v)^2

【検知性能指標】（正解ラベルは評価専用。検知器には渡さない）
  roc_auc / average_precision / recall_at_k / precision_at_k
"""

from typing import Dict

import numpy as np

# 検知器の識別子（ログCSVの列名・解析スクリプトと共有する）
DETECTOR_KEYS = ("d1_norm", "d2_cosine", "d3_kmeans", "d4_spectral")


# ============================================================
# 1) 検知器（更新ベクトル g_i を入力にスコアを返す）
# ============================================================
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def norm_deviation_scores(updates: np.ndarray) -> np.ndarray:
    """D1: 各更新のノルムと，全更新のノルム中央値との差の絶対値。"""
    norms = np.linalg.norm(updates, axis=1)
    return np.abs(norms - np.median(norms))


def cosine_distance_scores(updates: np.ndarray, reference: str = "median") -> np.ndarray:
    """
    D2: 参照方向 r とのコサイン距離 1 - cos(g_i, r)。
    reference="median" なら座標ごとの中央値，"mean" なら平均を参照方向に使う。
    """
    if reference == "median":
        r = np.median(updates, axis=0)
    elif reference == "mean":
        r = updates.mean(axis=0)
    else:
        raise ValueError(f"reference は 'median' か 'mean'（指定値: {reference!r}）")

    r_hat = _unit(r)
    if not np.any(r_hat):
        # 参照方向が定義できない（全更新が0など）場合は差をつけられない
        return np.zeros(updates.shape[0])

    norms = np.linalg.norm(updates, axis=1)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    cos = (updates @ r_hat) / norms_safe
    cos = np.where(norms > 1e-12, cos, 1.0)  # 0ベクトルは「異常なし」寄りに倒す
    return 1.0 - cos


def _pca_top2(centered: np.ndarray) -> np.ndarray:
    """中心化済み行列の第1・第2主成分への射影（列0=PC1, 列1=PC2）を返す。"""
    n = centered.shape[0]
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(2, vt.shape[0])
    proj = centered @ vt[:k].T  # (n, k)
    if k < 2:
        proj = np.hstack([proj, np.zeros((n, 2 - k))])
    return proj


def _kmeans2(x: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """
    2-means（Lloyd法）。初期中心は「最も離れた2点」に決め打ちして決定的にする。
    Returns: labels (n,) in {0,1}
    """
    n = x.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    centers = np.stack([x[i], x[j]])
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dist.argmin(axis=1)
        new_centers = np.stack([
            x[new_labels == c].mean(axis=0) if np.any(new_labels == c) else centers[c]
            for c in range(2)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            break
        labels, centers = new_labels, new_centers
    return labels


def kmeans_cluster_scores(updates: np.ndarray) -> np.ndarray:
    """
    D3: 特徴量 [ノルム, D2スコア, PC1, PC2] を標準化して 2-means。
    大きい方のクラスタを「正常」とみなし，その重心までの距離をスコアにする。
    """
    n = updates.shape[0]
    if n < 3:
        return np.zeros(n)

    centered = updates - updates.mean(axis=0)
    pc = _pca_top2(centered)
    feat = np.column_stack([
        np.linalg.norm(updates, axis=1),
        cosine_distance_scores(updates),
        pc[:, 0],
        pc[:, 1],
    ])
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (feat - mean) / std

    labels = _kmeans2(z)
    counts = np.bincount(labels, minlength=2)
    benign_label = int(np.argmax(counts))
    if counts[benign_label] == n:
        # クラスタが分かれなかった：全体重心からの距離で代用
        benign_center = z.mean(axis=0)
    else:
        benign_center = z[labels == benign_label].mean(axis=0)
    return np.linalg.norm(z - benign_center, axis=1)


def spectral_scores(updates: np.ndarray) -> np.ndarray:
    """D4: 中心化した更新行列の第1右特異ベクトル v への射影の2乗 (g_i . v)^2。"""
    n = updates.shape[0]
    if n < 2:
        return np.zeros(n)
    centered = updates - updates.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v = vt[0]
    return (centered @ v) ** 2


def all_detector_scores(updates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    D1--D4 のスコアをまとめて返す。
    updates は更新ベクトル g_i = w_i - w_global の (n_clients, dim) 配列。
    """
    updates = np.asarray(updates, dtype=np.float64)
    return {
        "d1_norm": norm_deviation_scores(updates),
        "d2_cosine": cosine_distance_scores(updates),
        "d3_kmeans": kmeans_cluster_scores(updates),
        "d4_spectral": spectral_scores(updates),
    }


# ============================================================
# 2) 検知性能指標（scikit-learn を使わず numpy だけで実装）
# ============================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    """同順位は平均順位にする rankdata（scipy.stats.rankdata の 'average' 相当）。"""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """
    ROC 曲線下面積。Mann-Whitney U 統計量に基づく閾値非依存の実装。
    正例（悪性）または負例（正常）が0個のときは nan を返す。
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, scores) -> float:
    """スコア降順に並べたときの，各悪性クライアント位置での precision の平均。"""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if int(y.sum()) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float(precision[y_sorted].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    """スコア上位 k 件を悪性と判定したときの悪性回収率（recall）。"""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = int(min(max(k, 0), len(s)))
    if k == 0:
        return 0.0
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / n_pos)


def precision_at_k(y_true, scores, k: int) -> float:
    """スコア上位 k 件のうち悪性が占める割合（precision@k）。"""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    k = int(min(max(k, 0), len(s)))
    if k == 0:
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / k)


def evaluate_scores(y_true, scores, k: int) -> Dict[str, float]:
    """1つの検知器スコアに対して主要指標をまとめて計算する。"""
    return {
        "auc": roc_auc(y_true, scores),
        "ap": average_precision(y_true, scores),
        "recall_at_k": recall_at_k(y_true, scores, k),
        "precision_at_k": precision_at_k(y_true, scores, k),
    }
