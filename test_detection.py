"""
test_detection.py

detection.py の単体テスト（research_plan_v2.tex 4.2--4.5 対応）
- 検知器 D1--D4：悪性更新が正常クラスタから外れているとき，各検知器が
  悪性クライアントに高いスコアを与える（AUC=1.0 になる）ことを確認する。
- ステルス攻撃（正常のばらつき内に収める）では AUC が 1.0 未満に落ちることを確認する。
- 検知性能指標 roc_auc / average_precision / recall_at_k / precision_at_k の
  基本的な性質（完全分離＝1.0，反転＝0.0，同順位の平均順位）を確認する。

実行：  python3 test_detection.py
"""

import numpy as np

from detection import (
    all_detector_scores,
    average_precision,
    cosine_distance_scores,
    kmeans_cluster_scores,
    norm_deviation_scores,
    precision_at_k,
    recall_at_k,
    roc_auc,
    spectral_scores,
    _rankdata,
)


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


# ============================================================
# ダミー更新ベクトルの生成
# ============================================================
def make_round(n_honest=16, n_malicious=4, dim=50, kind="obvious", seed=0):
    """
    honest は「共通の学習方向 g0 ＋ 小さなガウスノイズ」（実際のFLの更新に近い形）。
    malicious は kind に応じて動かす。
    Returns: deltas (n, dim), y_true (n,) with 1=malicious
    """
    rng = np.random.default_rng(seed)
    n = n_honest + n_malicious

    g0 = rng.normal(size=dim)
    g0 /= np.linalg.norm(g0)
    g0 *= 0.3  # 共通の学習方向（honest はこの周りに集まる）
    honest = g0 + rng.normal(scale=0.02, size=(n_honest, dim))

    if kind == "obvious":
        # 正常のばらつき（~0.02）よりはるかに大きく，学習方向と逆へ押し出す
        mal = -1.5 * g0 / np.linalg.norm(g0) + rng.normal(scale=0.02, size=(n_malicious, dim))
    elif kind == "stealth":
        # 正常の座標ごと平均・標準偏差から payload = mu - z*sigma（ALIE 相当）
        mu = honest.mean(axis=0)
        sigma = honest.std(axis=0)
        payload = mu - 1.0 * sigma
        mal = np.tile(payload, (n_malicious, 1))
    else:
        raise ValueError(kind)

    deltas = np.vstack([honest, mal])
    y = np.array([0] * n_honest + [1] * n_malicious)
    # シャッフルして位置に依存しないことを確認できるようにする
    perm = rng.permutation(n)
    return deltas[perm], y[perm]


# ============================================================
# 検知性能指標
# ============================================================
def test_roc_auc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    check("完全分離で AUC=1.0", roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0)
    check("完全逆順で AUC=0.0", roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0)
    check("全部同点で AUC=0.5", abs(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12)


def test_roc_auc_degenerate():
    check("正例が0個なら nan", np.isnan(roc_auc(np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]))))
    check("負例が0個なら nan", np.isnan(roc_auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]))))


def test_rankdata_ties():
    r = _rankdata(np.array([10.0, 20.0, 20.0, 40.0]))
    check("同順位は平均順位（2,3 -> 2.5）", np.allclose(r, [1.0, 2.5, 2.5, 4.0]), detail=str(r))


def test_average_precision_and_recall():
    y = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.05])  # 悪性2件がスコア上位2件
    check("悪性が上位に固まれば AP=1.0", abs(average_precision(y, scores) - 1.0) < 1e-12)
    check("recall@2 = 1.0", abs(recall_at_k(y, scores, 2) - 1.0) < 1e-12)
    check("precision@2 = 1.0", abs(precision_at_k(y, scores, 2) - 1.0) < 1e-12)
    check("recall@1 = 0.5", abs(recall_at_k(y, scores, 1) - 0.5) < 1e-12)


# ============================================================
# 検知器：分かりやすい攻撃なら AUC=1.0
# ============================================================
def test_detectors_separate_obvious_attack():
    deltas, y = make_round(kind="obvious", seed=1)
    for name, fn in [
        ("D1 norm", norm_deviation_scores),
        ("D2 cosine", cosine_distance_scores),
        ("D3 kmeans", kmeans_cluster_scores),
        ("D4 spectral", spectral_scores),
    ]:
        auc = roc_auc(y, fn(deltas))
        check(f"{name} は明白な攻撃を完全分離（AUC=1.0）", auc == 1.0, detail=f"AUC={auc:.3f}")


def test_all_detector_scores_keys_and_shape():
    deltas, y = make_round(kind="obvious", seed=2)
    scores = all_detector_scores(deltas)
    check("4検知器すべてのキーがある", set(scores) == {"d1_norm", "d2_cosine", "d3_kmeans", "d4_spectral"})
    for k, v in scores.items():
        check(f"{k} は (n,) 形状", v.shape == (deltas.shape[0],))


# ============================================================
# 検知器：ステルス攻撃なら検知性能が落ちる
# ============================================================
def test_stealth_attack_is_harder():
    deltas, y = make_round(kind="stealth", seed=3)
    aucs = {name: roc_auc(y, s) for name, s in all_detector_scores(deltas).items()}
    worst = min(aucs.values())
    check(
        "ステルス攻撃では少なくとも1つの検知器の AUC が 1.0 未満に落ちる",
        worst < 1.0,
        detail=str({k: round(v, 3) for k, v in aucs.items()}),
    )


def test_no_malicious_does_not_crash():
    rng = np.random.default_rng(0)
    deltas = rng.normal(scale=0.02, size=(12, 40))
    scores = all_detector_scores(deltas)
    y = np.zeros(12)
    for k, s in scores.items():
        check(f"{k}: 悪性0人でも nan の AUC を返すだけ（例外なし）", np.isnan(roc_auc(y, s)))


if __name__ == "__main__":
    tests = [
        test_roc_auc_perfect_and_reversed,
        test_roc_auc_degenerate,
        test_rankdata_ties,
        test_average_precision_and_recall,
        test_detectors_separate_obvious_attack,
        test_all_detector_scores_keys_and_shape,
        test_stealth_attack_is_harder,
        test_no_malicious_does_not_crash,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
