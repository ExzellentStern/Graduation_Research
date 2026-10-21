"""
test_detection.py

detection.py の単体テスト（sotsuron_plan2.tex 4.2--4.5 対応）
- 検知器 D1--D4：悪性更新が正常クラスタから外れているとき，各検知器が
  悪性クライアントに高いスコアを与える（AUC=1.0 になる）ことを確認する。
- ステルス攻撃（正常のばらつき内に収める）では AUC が 1.0 未満に落ちることを確認する。
- 検知性能指標 roc_auc / average_precision / recall_at_k / precision_at_k の
  基本的な性質（完全分離＝1.0，反転＝0.0，同順位の平均順位）を確認する。

実行：  python3 test_detection.py
"""

import numpy as np

from fl_detection.detection import (
    all_detector_scores,
    average_precision,
    cosine_distance_scores,
    filtered_mean_distance,
    kmeans_cluster_scores,
    norm_deviation_scores,
    precision_at_k,
    recall_at_k,
    roc_auc,
    spectral_scores,
    structure_diagnostics,
    STRUCTURE_KEYS,
    _kmeans2,
    _lloyd,
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


# ============================================================
# D1：偶数台のとき，真ん中の2台は数式上必ず同点（detection.norm_deviation_scores のメモ参照）
# ============================================================
def test_d1_middle_pair_is_exact_tie_for_even_n():
    rng = np.random.default_rng(0)
    naive_broken = 0
    n_trials = 500
    for _ in range(n_trials):
        updates = rng.normal(size=(20, 50))
        norms = np.linalg.norm(updates, axis=1)
        order = np.argsort(norms)
        lo, hi = order[9], order[10]  # ノルムの順で真ん中の2台

        naive = np.abs(norms - np.median(norms))
        naive_broken += naive[lo] != naive[hi]

        scores = norm_deviation_scores(updates)
        assert scores[lo] == scores[hi], "真ん中の2台のスコアが一致していない"
        # そろえても値そのものは変えていない（ずれは丸め誤差の範囲）
        assert np.allclose(scores, naive, rtol=0, atol=1e-9 * np.median(norms))
    check("偶数台では真ん中の2台の D1 スコアが完全に一致する", True, f"({n_trials} 回)")
    # このテストに意味があること（素直な計算では丸め誤差で同点が崩れること）の確認
    check("素直な計算では同点が崩れる場合がある", naive_broken > 0, f"({naive_broken}/{n_trials} 回)")


def test_d1_tie_counts_as_half_in_auc():
    # ノルムが 2, 4, 10, 20 の4台。中央値は 7 で，4 と 10 のスコアはどちらも 3（同点）。
    updates = np.array([[2.0, 0.0], [4.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    scores = norm_deviation_scores(updates)
    check("例（2, 4, 10, 20）でスコアは 5, 3, 3, 13", np.allclose(scores, [5.0, 3.0, 3.0, 13.0]), f"{scores}")
    # ノルム 4 のクライアントだけが悪性：良性は 5, 3, 13。3 との同点は 0.5 と数える -> (0 + 0.5 + 0) / 3
    auc = roc_auc([0, 1, 0, 0], scores)
    check("同点は「半分正解」として AUC に入る", abs(auc - 0.5 / 3) < 1e-12, f"AUC={auc}")


def test_d1_keeps_real_differences():
    # 差が小さくても，丸め誤差より十分大きければ同点にしない
    updates = np.array([[1.0, 0.0], [2.0, 0.0], [2.000001, 0.0], [5.0, 0.0], [9.0, 0.0]])
    scores = norm_deviation_scores(updates)  # 奇数台：中央値は 2.000001
    check("意味のある差（1e-6）はそのまま残る", scores[1] != scores[2] and len(set(scores.tolist())) == 5, f"{scores}")


# ============================================================
# D3：k-means は初期中心を全ての組で試し，最良の分割を返す（detection._kmeans2 のメモ参照）
# ============================================================
def test_kmeans2_tries_all_initial_pairs():
    import itertools

    rng = np.random.default_rng(3)
    worse_than_farthest = 0
    not_global = 0
    n_trials = 30
    for _ in range(n_trials):
        x = rng.normal(size=(10, 4))
        x[:3] += rng.normal(size=4) * 1.5  # 少数派のまとまりを1つ作る
        labels = _kmeans2(x)
        cost = sum(((x[labels == c] - x[labels == c].mean(axis=0)) ** 2).sum() for c in (0, 1))

        # 以前の方法（一番離れた2点から始める1回だけ）より悪くならない
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        _, cost_farthest = _lloyd(x, int(min(i, j)), int(max(i, j)))
        worse_than_farthest += cost > cost_farthest + 1e-9

        # 10点なら全ての分け方（511通り）を調べられるので，本当に最良かを確かめる
        best = min(
            sum(((x[np.array((0,) + b) == c] - x[np.array((0,) + b) == c].mean(axis=0)) ** 2).sum() for c in (0, 1))
            for b in itertools.product([0, 1], repeat=9) if sum(b) > 0
        )
        not_global += cost > best + 1e-9

        assert labels[0] == 0, "クライアント0のクラスタを 0 にそろえていない"
        assert np.array_equal(labels, _kmeans2(x)), "同じ入力で結果が変わった"
    check("以前の方法（最遠の2点から1回）より悪い分割にならない", worse_than_farthest == 0, f"({n_trials} 回)")
    check("全ての分け方の中で最良の分割になっている", not_global == 0, f"(最良でなかった回数 {not_global}/{n_trials})")


# ============================================================
# 参考の値：スコア上位を仮に取り除いたときの，残りの平均と良性の重心との距離
# ============================================================
def test_filtered_mean_distance():
    deltas, y = make_round(kind="obvious", seed=4)
    benign = np.where(y == 0)[0]
    f = int(y.sum())
    base = filtered_mean_distance(deltas, np.zeros(len(y)), benign, k=0)
    check("何も取り除かなければ，悪性のぶんだけ良性の重心からずれる", base > 0, f"dist={base:.4f}")

    perfect = filtered_mean_distance(deltas, y.astype(float), benign, k=f)   # 悪性にだけ高いスコア
    check("悪性をちょうど取り除けば，残りの平均は良性の重心に一致する", perfect < 1e-9, f"dist={perfect:.2e}")

    wrong = filtered_mean_distance(deltas, 1.0 - y, benign, k=f)             # 良性に高いスコア（逆向き）
    check("良性を取り除いて悪性を残すと，何もしないより悪くなる", wrong > base, f"wrong={wrong:.4f} base={base:.4f}")

    # 手計算と一致するか
    g = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    d = filtered_mean_distance(g, [0.1, 0.2, 0.9], benign_idx=[0, 1], k=1)   # 3台目を除く -> 残りの平均 (1,0) = 良性の重心
    check("手計算の例（取り除いた結果が良性の重心に一致）", abs(d) < 1e-12, f"d={d}")
    d0 = filtered_mean_distance(g, [0.1, 0.2, 0.9], benign_idx=[0, 1], k=0)  # 全員の平均 (4, 10/3) と (1,0) の距離
    check("手計算の例（何も除かない）", abs(d0 - np.hypot(3.0, 10.0 / 3.0)) < 1e-12, f"d0={d0}")


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


# ============================================================
# 分布構造の診断指標
# ============================================================
def test_structure_pc1_ratio_and_angle():
    rng = np.random.default_rng(0)
    n_honest, dim = 16, 30
    benign_idx = np.arange(n_honest)
    mal_idx = np.array([n_honest])

    # 正常更新は第0座標方向にだけ大きく広がる（細長い分布）
    honest = rng.normal(scale=0.01, size=(n_honest, dim))
    honest[:, 0] += rng.normal(scale=1.0, size=n_honest)
    center = honest.mean(axis=0)

    along = center.copy(); along[0] += 2.0          # 主要な変動方向に沿って押し出す
    ortho = center.copy(); ortho[1] += 2.0          # 直交方向へ押し出す
    d_along = structure_diagnostics(np.vstack([honest, along]), benign_idx, mal_idx)
    d_ortho = structure_diagnostics(np.vstack([honest, ortho]), benign_idx, mal_idx)

    check("細長い分布では pc1_ratio が 1 に近い", d_along["pc1_ratio"] > 0.95, f"{d_along['pc1_ratio']:.3f}")
    check("主成分方向への押し出しは |cos| ≒ 1", d_along["mal_cos_pc1"] > 0.99, f"{d_along['mal_cos_pc1']:.3f}")
    check("直交方向への押し出しは |cos| ≒ 0", d_ortho["mal_cos_pc1"] < 0.05, f"{d_ortho['mal_cos_pc1']:.3f}")

    iso = rng.normal(scale=1.0, size=(n_honest, dim))
    d_iso = structure_diagnostics(iso, benign_idx)
    check("等方的な分布では pc1_ratio が小さい", d_iso["pc1_ratio"] < 0.3, f"{d_iso['pc1_ratio']:.3f}")


def test_structure_rank_depends_on_distribution_shape():
    # 同じ mu - z*sigma でも，正常更新の分布形状によって「正常更新内での順位」が変わる
    rng = np.random.default_rng(1)
    n_honest, dim, z = 400, 200, 1.0
    benign_idx = np.arange(n_honest)
    mal_idx = np.array([n_honest])

    def rank_of_alie(honest):
        payload = honest.mean(axis=0) - z * honest.std(axis=0)
        return structure_diagnostics(np.vstack([honest, payload]), benign_idx, mal_idx)

    gauss = rank_of_alie(rng.normal(size=(n_honest, dim)))
    # 2群に分かれた正常更新（Non-IID で起こりうる多峰的な構造）
    sign = np.where(np.arange(n_honest)[:, None] < n_honest // 2, -1.0, 1.0)
    bimodal = rank_of_alie(sign + rng.normal(scale=0.1, size=(n_honest, dim)))

    check("正規分布なら順位は Phi(-1) ≒ 0.159 付近", abs(gauss["mal_rank_mean"] - 0.159) < 0.02,
          f"{gauss['mal_rank_mean']:.3f}")
    check("2峰分布では同じ z でも順位が Phi(-z) からずれる", abs(bimodal["mal_rank_mean"] - 0.159) > 0.05,
          f"{bimodal['mal_rank_mean']:.3f}")


def test_structure_without_malicious():
    rng = np.random.default_rng(2)
    d = structure_diagnostics(rng.normal(size=(12, 40)), np.arange(12))
    check("キーが STRUCTURE_KEYS と一致", set(d) == set(STRUCTURE_KEYS))
    check("悪性0人でも pc1_ratio は計算できる", np.isfinite(d["pc1_ratio"]))
    check("悪性0人なら悪性に関する指標は nan",
          all(np.isnan(d[k]) for k in ("mal_cos_pc1", "mal_cos_pc1_all", "mal_rank_mean", "mal_rank_std")))


if __name__ == "__main__":
    tests = [
        test_roc_auc_perfect_and_reversed,
        test_roc_auc_degenerate,
        test_rankdata_ties,
        test_average_precision_and_recall,
        test_detectors_separate_obvious_attack,
        test_d1_middle_pair_is_exact_tie_for_even_n,
        test_d1_tie_counts_as_half_in_auc,
        test_d1_keeps_real_differences,
        test_kmeans2_tries_all_initial_pairs,
        test_filtered_mean_distance,
        test_all_detector_scores_keys_and_shape,
        test_stealth_attack_is_harder,
        test_no_malicious_does_not_crash,
        test_structure_pc1_ratio_and_angle,
        test_structure_rank_depends_on_distribution_shape,
        test_structure_without_malicious,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
