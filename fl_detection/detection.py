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
                （偶数台では真ん中の2台が数式上必ず同点になる。丸め誤差で崩れないよう同じ値にそろえる。
                  経緯は norm_deviation_scores のメモを参照）
  D2 cosine   : 参照方向へのコサイン距離 s_i = 1 - cos(g_i, r),  r = coord-wise median
  D3 kmeans   : 標準化した特徴量 [norm, D2, PC1, PC2] への 2-means（初期中心を全ての2点の組で試し，最良の分割を使う）。
                大きい方のクラスタ（＝正常とみなす）重心までの距離。
  D4 spectral : 中心化した更新行列の第1右特異ベクトル v への射影の2乗 (g_i . v)^2

【検知性能指標】（正解ラベルは評価専用。検知器には渡さない）
  roc_auc / average_precision / recall_at_k / precision_at_k

【分布構造の診断指標】（正解ラベルを使う。検知器ではなく考察用の説明変数）
  structure_diagnostics : 第1主成分の寄与率 / 悪性方向と第1主成分の角度 /
                          悪性値の座標ごとの順位（正常更新内での経験分位）
"""

# 【このファイルの読み方】
#   検知に関する計算の「中身」だけを書いたファイル。attack.py と同じく，ここにある関数は
#   値を受け取って値を返すだけで，ファイルの読み書きも Flower とのやり取りもしない。
#   呼び出しているのは robust_strategy.py の aggregate_train()（毎ラウンド1回）。
#
#   中身は3つの部分に分かれている。
#       1) 検知器 D1〜D4     : 各クライアントの「悪性らしさ」のスコアを出す。正解（誰が悪性か）は使わない。
#       2) 分布構造の診断指標 : 良性の更新がどんな形で散らばり，悪性がその中のどこにいるかを測る。
#                               考察用で，正解を使う。検知器ではない。
#       3) 検知性能指標       : 検知器のスコアと正解を突き合わせて，ROC-AUC などの成績を付ける。
#
#   入力の形（どの関数もほぼ共通）
#       updates : 更新ベクトルの表。行 = クライアント（20台なら20行），列 = モデルの重み1つ1つ（約27万列）。
#                 値は「そのクライアントの学習でモデルがどれだけ動いたか」（届いた重み − 配った重み）。
#       updates[i] が i 番目（0から数える）のクライアントの更新ベクトル。
#
#   スコアの向き
#       4つの検知器とも「スコアが大きいほど悪性らしい」にそろえてある。
#       どの値から悪性と判定するか（しきい値）は決めていない。スコアで順位を付けるだけで，
#       その順位がどれだけ正解と合っているかを 3) の ROC-AUC で測る。
#
#   名前の先頭に _ が付いた関数（_unit など）は，このファイルの中だけで使う補助の関数。
#
# 【このファイルで何度も出てくる numpy の書き方】
#   np.linalg.norm(v)             : ベクトル v の長さ（ノルム）。各要素を2乗して合計し，平方根を取った値。
#   np.linalg.norm(表, axis=1)    : 行ごとに長さを計算する（クライアントごとに1つの値が出る）。
#   表.mean(axis=0)               : 列ごとに平均する（重み1つ1つについて，クライアント間の平均を取る）。
#   axis=0 は「行をまたいで（縦方向に）」，axis=1 は「列をまたいで（横方向に）」計算する，という意味。
#   A @ B                         : 行列の積（内積）。表 @ ベクトル なら，各行とベクトルの内積が並ぶ。
#   表[条件]                      : 条件が True の行だけを取り出す。
#   np.where(条件, A, B)          : 条件が成り立つ位置には A を，成り立たない位置には B を入れる。
#
# 【import の説明】
from typing import Dict, Optional, Sequence  # 型ヒント用の名前（辞書 / 「X か None」/ 順に並んだもの）

import numpy as np  # 数値計算のライブラリ（このファイルが使う外部ライブラリはこれだけ）

# 検知器の識別子（ログCSVの列名・解析スクリプトと共有する）
#   ( ... ) で囲んで並べたものはタプル（中身を変えられないリスト）。名前の一覧として使っている。
DETECTOR_KEYS = ("d1_norm", "d2_cosine", "d3_kmeans", "d4_spectral")

# 分布構造の診断指標の識別子（同上）
STRUCTURE_KEYS = (
    "pc1_ratio",        # 正常更新のみの第1主成分寄与率 lambda_1 / sum_k lambda_k
    "mal_cos_pc1",      # 悪性方向 a = g_mal - g_bar_B と，正常更新の第1主成分との |cos|
    "pc1_ratio_all",    # 全更新（悪性込み）での第1主成分寄与率（D4 が実際に見る行列）
    "mal_cos_pc1_all",  # 悪性方向 a と，全更新の第1主成分（D4 の射影方向）との |cos|
    "mal_rank_mean",    # 悪性値以下の正常更新の割合 r_j / |B| の座標平均
    "mal_rank_std",     # 同，座標間の標準偏差
)


# ============================================================
# 1) 検知器（更新ベクトル g_i を入力にスコアを返す）
# ============================================================
# 【_unit の説明】
#   ベクトルを，向きはそのままで長さ1にそろえる（単位ベクトルにする）補助関数。
#   「向き」だけを比べたいときに使う。
#     n = np.linalg.norm(v)  : v の長さ。
#     v / n if n > eps else np.zeros_like(v)
#                            : 長さが eps（0.000000000001）より大きければ，長さで割って長さ1にする。
#                              ほぼ0のベクトルは向きが決まらない（0で割れない）ので，全部0のベクトルを返す。
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


# D1 で「同点」とみなす差の上限（ノルムの中央値に対する比）。下の【メモ】を参照。
# 計算の丸め誤差はノルムの 1e-16 倍程度なので，それより十分大きく，
# かつ意味のある差（実際のスコアの差）よりはるかに小さい値にしてある。
D1_TIE_REL_TOL = 1e-9


# 【_snap_ties の説明】（D1 で使う補助。なぜ必要かは下の norm_deviation_scores のメモを参照）
#   order = np.argsort(scores, kind="mergesort")
#       : 「小さい順に並べたとき，もとの何番目の値が来るか」の一覧。値そのものではなく位置を返す。
#   snapped = scores[order].copy() : 小さい順に並べた値の写し。
#   for k in range(1, len(snapped)): : 2番目の値から順に，1つ前の値と比べる。
#       差が tol 以下なら，1つ前の値と同じ値に書き換える。
#   out = np.empty_like(scores); out[order] = snapped
#       : 並べ替える前の順番（クライアントの順）に戻す。
def _snap_ties(scores: np.ndarray, tol: float) -> np.ndarray:
    """差が tol 以下しかない値を，同じ値（小さい方）にそろえて返す。"""
    order = np.argsort(scores, kind="mergesort")
    snapped = scores[order].copy()
    for k in range(1, len(snapped)):
        if snapped[k] - snapped[k - 1] <= tol:
            snapped[k] = snapped[k - 1]
    out = np.empty_like(scores)
    out[order] = snapped
    return out


# 【D1（ノルム偏差）の説明】
#   考え方：更新の「大きさ」が，ほかのクライアントと違う者を疑う。
#     1. 各クライアントの更新ベクトルの長さ（ノルム）を計算する。
#     2. その中央値（大きさの順で真ん中の値）を求める。
#     3. 中央値からどれだけ離れているか（差の絶対値）をスコアにする。
#   極端に大きい（または小さい）更新を送る攻撃には強い。
#   大きさが良性と変わらない攻撃（符号反転は大きさが同じ，ALIE は良性の範囲内）には弱い。
#
#   np.median(norms)        : 中央値。平均ではなく中央値を使うのは，悪性の極端な値に引っ張られにくいため。
#   np.abs(norms - median)  : 差の絶対値（マイナスを取る）。大きくても小さくても「離れている」と数える。
def norm_deviation_scores(updates: np.ndarray) -> np.ndarray:
    """
    D1: 各更新のノルムと，全更新のノルム中央値との差の絶対値。

    【メモ：同点の扱いについて（2026-10-09）】
    クライアント数が偶数のとき，中央値は「ノルムの順で真ん中の2台（a <= b）の平均」になる。
    すると，その2台のスコアはどちらも (b - a) / 2 で，a と b の値によらず数式の上で必ず等しい。
      例: ノルムが 2, 4, 10, 20 なら中央値は (4 + 10) / 2 = 7 で，4 と 10 のスコアはどちらも 3。
    これは D1 の計算方法が自分で作り出す同点であり，2台の更新が似ているから起きるわけではない。

    ところが，素直に |ノルム - 中央値| を計算すると，丸め誤差で
    3.0000000000000004 と 2.9999999999999996 のようにわずかにずれ，同点でなくなることがある。
    その2台が「悪性と良性」の組だった場合，意味のない丸め誤差で順位が決まり，ROC-AUC が動いてしまう
    （20台・悪性4台で最大 0.03 程度。試運転では攻撃ありのラウンドの3回に1回ほどで該当した）。

    等しいはずの値は等しく扱うべきなので，差が丸め誤差としか考えられないほど小さい
    （ノルム中央値の D1_TIE_REL_TOL 倍以下）スコアは，同じ値にそろえてから返す。
    こうすると roc_auc() は同点を「半分正解」として数える（_rankdata が同順位を平均順位にするため）。
    そろえるのは D1 だけ。D2--D4 には，このような構造的な同点は生じない。
    """
    norms = np.linalg.norm(updates, axis=1)
    median = np.median(norms)
    return _snap_ties(np.abs(norms - median), tol=D1_TIE_REL_TOL * median)


# 【D2（コサイン距離）の説明】
#   考え方：更新の「向き」が，みんなの代表的な向きとずれている者を疑う。
#     1. 代表的な向き（参照方向）r を決める。重み1つ1つについてクライアント間の中央値を取ったベクトル。
#     2. 各クライアントの更新ベクトルと r のなす角を，コサイン（cos）で測る。
#          同じ向きなら cos = 1，直角なら 0，正反対なら -1。
#     3. スコア = 1 - cos。同じ向きなら 0，直角なら 1，正反対なら 2。
#   向きを逆にする符号反転には強い（正反対なのでスコアが最大になる）。大きさは見ていない。
#
#   np.median(updates, axis=0)  : 参照方向 r。列ごと（重み1つ1つについて）の，クライアント間の中央値。
#   r_hat = _unit(r)            : 参照方向を長さ1にする。
#   if not np.any(r_hat):       : np.any は「0でない要素が1つでもあるか」。全部0なら参照方向が決まらないので，
#                                 全員のスコアを0にして返す。
#   cos = (updates @ r_hat) / norms_safe
#       : 内積を長さで割ると cos になる（cos = 内積 ÷（長さ × 長さ）。r_hat の長さは1）。
#   norms_safe / np.where(...)  : 長さがほぼ0の更新は0で割れないので，割る数を1に置き換えておき，
#                                 あとで cos を 1（＝スコア0，異常なし）に直している。
def cosine_distance_scores(updates: np.ndarray) -> np.ndarray:
    """
    D2: 参照方向 r とのコサイン距離 1 - cos(g_i, r)。
    参照方向 r は，座標ごとの中央値（coordinate-wise median）に固定している。

    【メモ：参照方向を中央値に固定したことについて（2026-10-09）】
    以前は引数 reference で "median"（中央値）と "mean"（平均）を選べたが，実験では常に中央値を
    使っており，平均は一度も使っていなかった。使わない選択肢と既定値がコードに残っていると
    紛らわしいので，選択肢をなくして中央値に固定した。
    中央値を使うのは，悪性クライアントの極端な値に参照方向が引っ張られにくいため
    （平均だと，悪性の更新そのものが参照方向を動かしてしまう）。
    平均を参照方向にした検知器を試したくなった場合は，下の np.median を updates.mean に変える。
    その場合は別の検知器として扱い，結果は中央値のものと混ぜないこと。
    """
    r = np.median(updates, axis=0)

    r_hat = _unit(r)
    if not np.any(r_hat):
        # 参照方向が定義できない（全更新が0など）場合は差をつけられない
        return np.zeros(updates.shape[0])

    norms = np.linalg.norm(updates, axis=1)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    cos = (updates @ r_hat) / norms_safe
    cos = np.where(norms > 1e-12, cos, 1.0)  # 0ベクトルは「異常なし」寄りに倒す
    return 1.0 - cos


# 【_pca_top2 の説明】（D3 で使う補助）
#   主成分分析（PCA）。約27万次元ある更新ベクトルを，「クライアント間の違いが一番よく表れる方向」2つに
#   絞って，2つの数値で表す。
#     第1主成分（PC1）: クライアントの更新が，一番大きくばらついている方向。
#     第2主成分（PC2）: PC1 と直角な方向のうち，次に大きくばらついている方向。
#   centered : 各列から平均を引いた表（「全体の中心からのずれ」にしたもの）。
#   np.linalg.svd(centered, full_matrices=False)
#       : 特異値分解。3つの値を返す。ここで使うのは3つ目の vt で，
#         vt[0] が PC1 の方向，vt[1] が PC2 の方向（どちらも長さ1のベクトル）。
#   centered @ vt[:k].T : 各クライアントの更新を，その方向に射影する（その方向の成分の大きさを取り出す）。
#   np.hstack / np.zeros : クライアントが少なく方向が1つしか取れないとき，2列目を0で埋める。
def _pca_top2(centered: np.ndarray) -> np.ndarray:
    """中心化済み行列の第1・第2主成分への射影（列0=PC1, 列1=PC2）を返す。"""
    n = centered.shape[0]
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(2, vt.shape[0])
    proj = centered @ vt[:k].T  # (n, k)
    if k < 2:
        proj = np.hstack([proj, np.zeros((n, 2 - k))])
    return proj


# 【_lloyd / _kmeans2 の説明】（D3 で使う補助）
#   k-means 法で，クライアントを2つのグループに分ける。
#
#   _lloyd(x, i, j) : 最初の中心を「i 番目の点」と「j 番目の点」にして，k-means を1回行う。
#     1. 各クライアントを，近いほうの中心のグループに入れる。
#     2. 各グループの平均を，新しい中心にする。
#     3. グループ分けが変わらなくなるまで 1 と 2 を繰り返す（最大 n_iter 回）。
#     返すのは（グループ分け, その分け方の「まとまりの悪さ」）。
#     まとまりの悪さ = 各点から，自分のグループの中心までの距離の2乗を，全部足したもの。
#     小さいほど，グループがよくまとまっている（k-means が小さくしようとしている量そのもの）。
#
#   _kmeans2(x) : 最初の中心の選び方を全部（2点の組み合わせすべて）試し，
#                 まとまりの悪さが一番小さかった分け方を返す。
#
#   dist.argmin(axis=1)           : 各クライアントについて，近いほうの中心の番号（0 か 1）。
#   np.array_equal / np.allclose  : 前回と同じ（ほぼ同じ）かどうかの判定。同じなら break で繰り返しを終える。
#   range(n) の2重の for          : i < j となる組（i, j）を，小さい順にすべて取り出す。
#
# 【メモ：最初の中心を全部の組み合わせで試すようにしたことについて（2026-10-09）】
#   k-means は，最初の中心の置き方によって，違う分け方に落ち着くことがある（最初の置き方が悪いと，
#   もっと良い分け方があるのに，そこへたどり着けない）。そのため，ふつうは最初の中心を変えて何度も試し，
#   一番良い分け方を採用する。
#
#   以前の実装は，実行のたびに結果が変わらないよう，最初の中心を「互いに一番離れている2点」の1通りに
#   固定していた。ところが調べてみると，この1通りでは，ほとんどの場合，一番良い分け方に届いていなかった。
#   人工的に作ったデータ（20台，悪性4台）で，全部の組み合わせを試した場合と比べた結果：
#       ALIE     : 一番良い分け方と一致したのは 30回中 0回。D3 の AUC は 0.817（全部試すと 0.848）。
#       符号反転 : 一番良い分け方と一致したのは 30回中 5回。D3 の AUC は 0.912（全部試すと 0.948）。
#   つまり，D3 の成績が，検知器の考え方の限界ではなく，分け方の探し方が不十分なせいで低く出ていた。
#   このままだと「D3 は Non-IID で崩れる」という結果が出ても，原因が検知器の考え方にあるのか，
#   探し方にあるのかを区別できない。
#
#   そこで，最初の中心を全部の組み合わせ（20台なら 190通り）で試し，一番良い分け方を採用するようにした。
#   乱数を使わないので，同じ入力なら必ず同じ結果になる（再現性は保たれる）。計算は一瞬で終わる。
#   「一番離れた2点」という決め方にも根拠はあった（離れた2点は別のグループになりやすい）が，
#   それは速く計算するための工夫であり，20台という少ない台数では，全部試すほうが確実である。
#
#   同じまとまりの悪さの分け方が複数あった場合は，先に試したほうを採用する（i, j の小さい順に試す）。
#   また，グループの番号（0 と 1）は付け替えても同じ分け方なので，
#   「0番のクライアントが入っているグループを 0 とする」と決めて，番号の付け方も毎回同じにしている。
def _lloyd(x: np.ndarray, i: int, j: int, n_iter: int = 100):
    """初期中心を x[i], x[j] とした 2-means（Lloyd法）。Returns: (labels (n,) in {0,1}, クラスタ内平方和)"""
    centers = np.stack([x[i], x[j]])
    labels = np.zeros(x.shape[0], dtype=int)
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
    cost = float(((x - centers[labels]) ** 2).sum())
    return labels, cost


def _kmeans2(x: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """
    2-means。初期中心を全ての2点の組で試し，クラスタ内平方和が最小の分割を返す（乱数を使わず決定的）。
    Returns: labels (n,) in {0,1}。クライアント0が属するクラスタを 0 とする。
    """
    n = x.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    best_labels, best_cost = None, float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            labels, cost = _lloyd(x, i, j, n_iter)
            # 丸め誤差程度の差では入れ替えない（同じ分け方が，試す順番によって選ばれたり選ばれなかったりしないように）
            if best_labels is None or cost < best_cost * (1.0 - 1e-12):
                best_labels, best_cost = labels, cost
    if best_labels[0] == 1:
        best_labels = 1 - best_labels
    return best_labels


# 【D3（k-means）の説明】
#   考え方：大きさ・向き・主な変動の方向をまとめて見て，「多数派のグループ」から遠い者を疑う。
#     1. 各クライアントを4つの数値（特徴量）で表す。
#          更新の大きさ / D2 のスコア（向きのずれ）/ PC1 の成分 / PC2 の成分
#     2. 4つの数値を標準化する（それぞれ平均0・標準偏差1にそろえる）。
#        大きさと向きでは値の桁が違うので，そろえないと桁の大きいものだけが効いてしまう。
#     3. k-means で2グループに分ける（最初の中心を全部の組み合わせで試し，一番良い分け方を使う）。
#     4. 人数の多いほうを「良性のグループ」とみなし，その中心までの距離をスコアにする。
#   D1 と D2 の情報を両方使うので，どちらか一方でしか見えない攻撃にも対応しやすい。
#   「多数派が良性」という前提に立っているので，悪性が多数を占めると成り立たない。
#
#   np.column_stack([...])       : 4つの配列を，列として横に並べて表にする（20行 × 4列）。
#   (feat - mean) / std          : 標準化。std がほぼ0の列は，0で割らないよう std を1に置き換えている。
#   np.bincount(labels, minlength=2) : 各グループの人数を数える。
#   np.argmax(counts)            : 人数が多いほうのグループの番号。
#   if counts[benign_label] == n : 全員が同じグループになった（分かれなかった）場合は，全体の中心からの距離を使う。
#   z[labels == benign_label]    : 良性グループに入ったクライアントの行だけを取り出す。
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


# 【D4（スペクトル）の説明】
#   考え方：クライアント間で一番大きくばらついている方向（第1主成分）に，大きく突き出している者を疑う。
#     1. 各列から平均を引いて，「全体の中心からのずれ」にする。
#     2. 一番ばらつきの大きい方向 v を求める（特異値分解の vt[0]）。
#     3. 各クライアントのずれを v に射影し，その2乗をスコアにする。
#   悪性クライアントがまとまって同じ方向にずれていると，その方向が「一番ばらつきの大きい方向」になりやすく，
#   悪性のスコアが大きくなる。良性のばらつきのほうが大きいと，v は良性どうしの違いを表す方向になり，
#   悪性を見つけられない（Non-IID が強いと，まさにこれが起きうる）。
#
#   centered @ v : 各クライアントのずれと v の内積（v 方向の成分）。
#   ** 2         : 2乗。プラス方向でもマイナス方向でも，離れていれば大きい値になる。
def spectral_scores(updates: np.ndarray) -> np.ndarray:
    """D4: 中心化した更新行列の第1右特異ベクトル v への射影の2乗 (g_i . v)^2。"""
    n = updates.shape[0]
    if n < 2:
        return np.zeros(n)
    centered = updates - updates.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v = vt[0]
    return (centered @ v) ** 2


# 【all_detector_scores の説明】
#   4つの検知器をまとめて実行する。robust_strategy.py が呼んでいるのはこの関数。
#   返すのは辞書で，名前（"d1_norm" など）-> その検知器の全クライアントぶんのスコア。
#     例: scores["d2_cosine"][3] が，D2 の，3番目（0から数える）のクライアントのスコア。
#   np.asarray(updates, dtype=np.float64) : 入力を小数（倍精度）の numpy 配列にそろえる。
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
# 2) 分布構造の診断指標（spread の「大きさ」では捉えられない「形」を測る）
# ============================================================
# 【この部分の説明】
#   検知器ではなく，「なぜ検知できた／できなかったのか」を考えるための指標。
#   良性の更新の散らばり方には「大きさ」（spread）のほかに「形」がある。
#   同じ大きさでも，丸く均等に散らばっているのか，ある方向に細長く伸びているのかで，
#   悪性の見つけやすさは変わる。その「形」と，悪性がその中のどこにいるかを数値にする。
#   ここでは正解（誰が悪性か）を使う。結果は検知器には渡さず，ログに記録するだけ。
#
# 【_pc1 の説明】（補助）
#   中心化した表について，（第1主成分の寄与率, 第1主成分の方向）の2つを返す。
#     寄与率 = 一番ばらつきの大きい方向が，ばらつき全体のうち何割を占めるか。
#              1 に近い     -> ほとんどのばらつきが1つの方向に集中している（細長い形）。
#              小さい       -> いろいろな方向に均等に散らばっている（丸い形）。
#   s は特異値（各方向のばらつきの大きさを表す値）。s ** 2 の合計が，ばらつき全体にあたる。
#   行が2つ未満，またはばらつきがほぼ0のときは計算できないので，(nan, None) を返す。
#   float("nan") は「数値として定義できない」ことを表す特別な値（Not a Number）。
def _pc1(centered: np.ndarray):
    """中心化済み行列の（第1主成分の寄与率, 第1右特異ベクトル）。定義できなければ (nan, None)。"""
    if centered.shape[0] < 2:
        return float("nan"), None
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    total = float((s ** 2).sum())
    if total <= 1e-24:
        return float("nan"), None
    return float(s[0] ** 2 / total), vt[0]


# 【structure_diagnostics の説明】
#   6つの診断指標（STRUCTURE_KEYS）を計算して，辞書で返す。
#     pc1_ratio       : 良性だけで見た，第1主成分の寄与率（良性の散らばりが細長いか丸いか）。
#     pc1_ratio_all   : 悪性も含めた全員で見た寄与率（D4 が実際に見ている表）。
#     mal_cos_pc1     : 悪性のずれている方向と，良性の第1主成分の方向が，どれだけ重なっているか（0〜1）。
#                       1 に近い -> 悪性は，良性がもともと大きくばらついている方向にずれている（紛れやすい）。
#                       0 に近い -> 悪性は，良性があまりばらつかない方向にずれている（目立ちやすい）。
#     mal_cos_pc1_all : 同じく，全員で見た第1主成分（D4 が見る方向）との重なり。
#                       1 に近いほど，D4 が見ている方向が悪性の方向と一致している。
#     mal_rank_mean   : 重み1つ1つについて「悪性の値以下である良性が何割いるか」を求め，全部の重みで平均した値。
#                       0.5 なら悪性は良性の真ん中あたり，0 や 1 に近いと良性の端より外側にいる。
#     mal_rank_std    : その割合が，重みによってどれだけばらつくか。
#   悪性がいない場合（攻撃なしの条件），悪性に関する4つは nan になる。
#
#   out = {key: float("nan") for key in STRUCTURE_KEYS}
#       : 辞書内包表記。6つの名前すべてに nan を入れた辞書を最初に作り，計算できたものだけ後で上書きする。
#   a_hat = _unit(malicious.mean(axis=0) - benign_center)
#       : 悪性の平均の位置から良性の重心を引いた「悪性のずれている方向」を，長さ1にしたもの。
#   abs(a_hat @ v1) : 2つの方向の重なり具合（cos の絶対値）。向きが逆でも同じ直線上なら1になる。
#   (benign <= m[None, :]).mean(axis=0)
#       : 重み1つ1つについて，「悪性の値 m 以下である良性クライアントの割合」を求める。
#         benign <= m は True/False の表になり，mean を取ると True の割合になる。
#   [... for m in malicious] : 悪性クライアント1台ずつについて同じ計算をする（リスト内包表記）。
def structure_diagnostics(
    updates: np.ndarray,
    benign_idx: Sequence[int],
    malicious_idx: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    正常更新の分布構造と，その中での悪性更新の位置を要約する（STRUCTURE_KEYS）。

    - pc1_ratio / pc1_ratio_all : 中心化した更新行列の lambda_1 / sum_k lambda_k。
      1 に近いほど特定方向へ細長く，1/(n-1) に近いほど等方的に広がっている。
    - mal_cos_pc1 / mal_cos_pc1_all : 悪性方向 a = mean_M(g) - g_bar_B と第1主成分 v1 の
      |cos|。1 なら主要な変動方向に沿い，0 なら直交方向へ外れている。
    - mal_rank_mean / mal_rank_std : 各座標 j で r_j = #{i in B : g_ij <= g_mal_j} を数え，
      r_j / |B| を座標間で要約したもの（悪性が複数なら悪性間でも平均）。
      正常更新が座標ごとに正規分布なら，ALIE（mu - z*sigma）では Phi(-z) 付近に集中する。

    悪性がいない場合，悪性に関する指標は nan。
    """
    updates = np.asarray(updates, dtype=np.float64)
    benign_idx = np.asarray(benign_idx, dtype=int)
    malicious_idx = np.asarray([] if malicious_idx is None else malicious_idx, dtype=int)
    out = {key: float("nan") for key in STRUCTURE_KEYS}
    if len(benign_idx) == 0:
        return out

    benign = updates[benign_idx]
    benign_center = benign.mean(axis=0)
    out["pc1_ratio"], v1 = _pc1(benign - benign_center)
    out["pc1_ratio_all"], v1_all = _pc1(updates - updates.mean(axis=0))
    if len(malicious_idx) == 0:
        return out

    malicious = updates[malicious_idx]
    a_hat = _unit(malicious.mean(axis=0) - benign_center)
    if np.any(a_hat):
        if v1 is not None:
            out["mal_cos_pc1"] = float(abs(a_hat @ v1))
        if v1_all is not None:
            out["mal_cos_pc1_all"] = float(abs(a_hat @ v1_all))

    # 悪性1人ずつ処理して (|M|, |B|, dim) の巨大な中間配列を作らない
    rank_frac = np.mean(
        [(benign <= m[None, :]).mean(axis=0) for m in malicious], axis=0
    )
    out["mal_rank_mean"] = float(rank_frac.mean())
    out["mal_rank_std"] = float(rank_frac.std())
    return out


# ============================================================
# 3) 検知性能指標（scikit-learn を使わず numpy だけで実装）
# ============================================================
# 【この部分の説明】
#   検知器の成績を付ける部分。入力はどの関数も同じ。
#       y_true : 正解。クライアントごとに，悪性なら 1，良性なら 0。
#       scores : 検知器が出したスコア（大きいほど悪性らしい）。
#   正解を使うのはここ（と 2) の診断指標）だけ。検知器 D1〜D4 は正解を知らずにスコアを出している。
#
# 【_rankdata の説明】（ROC-AUC で使う補助）
#   値に，小さい順の順位（1位，2位，…）を付ける。同じ値どうしには，順位の平均を付ける。
#     例: [10, 20, 20, 30] -> [1, 2.5, 2.5, 4]   （20 が2つあるので，2位と3位の平均の 2.5）
#   この「同順位は平均」という扱いのおかげで，ROC-AUC で同点が「半分正解」として数えられる。
#   while の中は，同じ値が続いている範囲（i から j まで）を探して，その範囲に平均の順位を入れている。
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


# 【roc_auc の説明】（この研究の主な指標）
#   ROC-AUC。意味は次のとおり。
#     悪性クライアントを1台，良性クライアントを1台，でたらめに選んだとき，
#     悪性のほうがスコアが高くなっている確率。
#       1.0 : どの組でも悪性のほうが高い（完全に見分けられる）。
#       0.5 : 半々（でたらめに順位を付けたのと同じ。見分けられていない）。
#       0.0 : どの組でも悪性のほうが低い（悪性のほうが「良性らしい」と判定されている）。
#   しきい値を決めなくても計算できるので，検知器どうしを公平に比べられる。
#
#   計算は，全部の組を1つずつ比べる代わりに，順位を使って一度に求めている（Mann-Whitney の U 統計量）。
#     y = ...astype(bool)   : 正解を True/False にする（True が悪性）。
#     n_pos / n_neg         : 悪性の台数 / 良性の台数。~y は True と False を入れ替えたもの。
#     どちらかが0台なら「悪性と良性の組」が作れないので，nan を返す（攻撃なしの条件はこれ）。
#     ranks[y].sum()        : 悪性クライアントの順位の合計。
#     そこから「悪性どうしで付く順位のぶん」を引き，組の総数（悪性の台数 × 良性の台数）で割る。
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


# 【average_precision の説明】
#   スコアの高い順に並べて上から見ていき，悪性クライアントが現れるたびに
#   「ここまで見た中で，悪性は何割か」（precision）を記録し，その平均を取る。
#   悪性が上位に固まっているほど 1 に近づく。
#     np.argsort(-s, kind="mergesort") : スコアの高い順に並べたときの位置（マイナスを付けて降順にしている）。
#     np.cumsum(y_sorted)              : 累積和。上から k 番目までに悪性が何台いたか。
#     tp / (番号 + 1)                  : 上から k 番目までの中の，悪性の割合。
#     precision[y_sorted].mean()       : 悪性が現れた位置での値だけを取り出して平均する。
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


# 【recall_at_k / precision_at_k の説明】
#   「スコアの高い順に上位 k 台を悪性と判定する」と決めたときの成績。
#   実験では k に悪性の台数（20台中4台なら4）を渡している。
#     recall_at_k    : 本当の悪性のうち，上位 k 台に入っていた割合（見逃さなかった割合）。
#     precision_at_k : 上位 k 台のうち，本当に悪性だった割合（誤って疑わなかった割合）。
#   k が悪性の台数と同じとき，この2つは同じ値になる。
#     min(max(k, 0), len(s))  : k を 0 以上・クライアント数以下の範囲に収める。
#     np.argsort(-s, ...)[:k] : スコアの高い順に並べたときの，上位 k 台の位置。
#     y[top].sum()            : その中にいる悪性の台数。
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


# 【filtered_mean_distance の説明】
#   「検知器の言うとおりに取り除いていたら，平均はどれだけ正しくなっていたか」を測る。
#     1. スコアの高い順に，上位 k 台を取り除く（k には悪性の台数を渡す）。
#     2. 残ったクライアントの更新を，単純に平均する。
#     3. その平均が，本来の良性の重心（良性だけの平均）から，どれだけ離れているかを返す。
#   値が小さいほど，取り除いた結果が「良性だけの平均」に近い。
#     0 に近い       : 悪性をちょうど取り除けた（残ったのは良性だけ）。
#     何も除かない場合（k=0）の値より大きい : 悪性を残して良性を取り除いてしまい，かえって悪くなった。
#   実際に取り除くわけではなく，仮に計算してみるだけである。集約には影響しない。
#
#   np.argsort(-scores, kind="mergesort")[k:] : スコアの高い順に並べ，上位 k 台を除いた残りの位置。
#   updates[keep].mean(axis=0)                : 残ったクライアントの更新の平均。
#
# 【メモ：この指標は「記録するだけ」にしていることについて（2026-10-09）】
#   研究計画書には当初，「防御効用（橋渡し指標）」として，検知性能と並ぶ評価指標に挙げられていた
#   （検知できることと，実際に防御できることをつなぐ，という位置づけ）。
#   その後，本研究は「検知できるかどうか」に絞ることにしたため，計画書の評価指標からは外した。
#   卒業論文では扱わない。
#   それでもログに残しているのは，この値がスイープの後からは計算できないからである
#   （計算には更新ベクトルそのものが必要だが，大きすぎるので保存していない）。
#   計算は1ラウンドあたり 0.01 秒未満で，スイープ全体でも 20 秒ほどしか増えない。
#   後で必要になったときに，10 時間かけて回し直さずに済むよう，記録だけしてある。
#
#   使う場合の注意：
#     - ここでの平均は「単純な平均」だが，実際の集約（FedAvg）はデータ数で重み付けした平均である。
#       そのため，この値は「実際に防御したときの効果」そのものではない。
#     - 値の良し悪しは，何も取り除かなかった場合の値（k=0 で計算したもの）と比べて読む。
def filtered_mean_distance(updates: np.ndarray, scores, benign_idx: Sequence[int], k: int) -> float:
    """
    スコア上位 k 件を除外して残りを単純平均した g_hat と，正常更新の重心 g_bar_B との距離 ||g_hat - g_bar_B||。
    正常クライアントがいない場合は nan。
    """
    updates = np.asarray(updates, dtype=np.float64)
    benign_idx = np.asarray(benign_idx, dtype=int)
    if len(benign_idx) == 0:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    k = int(min(max(k, 0), len(s) - 1))  # 全員を取り除くことはしない
    keep = np.argsort(-s, kind="mergesort")[k:]
    return float(np.linalg.norm(updates[keep].mean(axis=0) - updates[benign_idx].mean(axis=0)))


# 【evaluate_scores の説明】
#   上の4つの指標をまとめて計算し，辞書で返す。robust_strategy.py が検知器ごとに呼んでいる。
def evaluate_scores(y_true, scores, k: int) -> Dict[str, float]:
    """1つの検知器スコアに対して主要指標をまとめて計算する。"""
    return {
        "auc": roc_auc(y_true, scores),
        "ap": average_precision(y_true, scores),
        "recall_at_k": recall_at_k(y_true, scores, k),
        "precision_at_k": precision_at_k(y_true, scores, k),
    }


# ============================================================
# 【補足メモ】このファイルの難しい箇所について（2026-10-09）
#
# 上の各関数の説明では，「何を求めているか」と「返ってくる値のどれを使っているか」だけを書き，
# 数学の中身には踏み込んでいない。ここに，その先を知りたくなったときのための補足をまとめておく。
# 実験を回したり結果を読んだりするだけなら，ここは読まなくてよい。
# ============================================================
#
# ------------------------------------------------------------
# (1) 主成分分析（PCA）と特異値分解（SVD）  ->  _pca_top2 / spectral_scores / _pc1
# ------------------------------------------------------------
# 知りたいこと
#   クライアントの更新ベクトルは約27万個の数値でできている。その中で，
#   「クライアントによる違いが一番はっきり出る方向」を見つけたい。
#
# たとえ
#   教室の生徒の「身長」と「体重」を，横軸・縦軸に取って点を打つと，点は右上がりの細長い雲になる。
#   その雲が一番長く伸びている向き（「体が大きい／小さい」にあたる向き）が第1主成分，
#   それと直角で，次に広がっている向き（「太め／細め」にあたる向き）が第2主成分。
#   ここでは，その「身長・体重」の2個が，約27万個の重みに増えているだけである。
#
# 手順
#   1. 中心化：各列（重み1つ1つ）から，その列の平均を引く。
#      「どこにいるか」ではなく「全体の中心からどっちにどれだけずれているか」を見るため。
#   2. 特異値分解：中心化した表 X を，3つの部品に分ける。
#          U, s, Vt = np.linalg.svd(X, full_matrices=False)
#        Vt の各行 : ばらつきの大きい順に並んだ「方向」（長さ1のベクトル）。Vt[0] が第1主成分の方向。
#        s         : 各方向の「ばらつきの大きさ」を表す値（特異値）。大きい順に並んでいる。
#        U         : 各クライアントが，各方向にどれだけずれているかの情報（このファイルでは使っていない）。
#      クライアントが20台なら，取り出せる方向は最大で20個（行数ぶん）しかない。
#      full_matrices=False は，その必要なぶんだけを計算する指定（27万×27万の表を作らずに済む）。
#   3. 射影：X @ Vt[0] で，各クライアントが第1主成分の方向にどれだけずれているかが1つの数値で出る。
#
# 寄与率
#   s[0] ** 2 / (s ** 2).sum()  が，第1主成分の寄与率。
#   「ばらつき全体のうち，一番大きい方向が何割を占めるか」を表す。
#   各方向のばらつき（分散）は s の2乗に比例するので，2乗してから割合を取っている。
#
# D4 との関係
#   D4 のスコアは (X @ Vt[0]) ** 2，つまり「第1主成分の方向へのずれの2乗」。
#   悪性がまとまって同じ方向にずれていれば，その方向が第1主成分になり，悪性のスコアが大きくなる。
#   良性どうしの違いのほうが大きければ，第1主成分は良性の違いを表す方向になり，悪性は目立たない。
#   どちらになっているかを確かめるための指標が mal_cos_pc1_all（悪性の方向と第1主成分の重なり）。
#
# ------------------------------------------------------------
# (2) k-means 法  ->  _kmeans2 / kmeans_cluster_scores
# ------------------------------------------------------------
# 知りたいこと
#   クライアントを，似た者どうしの2つのグループに分けたい。
#
# 手順（グループの「中心」と「所属」を，交互に決め直す）
#   1. 中心を2つ，仮に決める。
#   2. 各クライアントを，近いほうの中心のグループに入れる。
#   3. 各グループの平均の位置を，新しい中心にする。
#   4. 所属が変わらなくなるまで 2 と 3 を繰り返す。
#   中心が動けば所属が変わり，所属が変われば中心が動く。これを繰り返すと，やがて動かなくなる。
#
# このファイルでの工夫
#   - 最初の中心の置き方によって，違う分け方に落ち着くことがある。乱数で決めると，実行のたびに結果も変わりうる。
#     ここでは，最初の中心を全部の組み合わせ（2点の組すべて）で試し，一番良い分け方を採用している。
#     乱数を使わないので，同じ入力なら必ず同じ結果になる（経緯は _kmeans2 のメモを参照）。
#   - 約27万次元のまま距離を測ると，ほとんどの成分は検知に関係のない雑音なので，
#     まず4つの特徴量（大きさ・向きのずれ・PC1・PC2）に要約してから分けている。
#   - 4つの特徴量は単位も桁も違うので，分ける前に標準化する。
#       標準化 = （値 − 平均）÷ 標準偏差。どの特徴量も「平均0・ばらつき1」にそろう。
#     これをしないと，たまたま桁の大きい特徴量だけで距離が決まってしまう。
#
# 前提
#   「人数の多いグループが良性」とみなしている。悪性が少数派である限り成り立つ（実験は20台中4台）。
#
# ------------------------------------------------------------
# (3) ROC-AUC を順位から求める計算  ->  _rankdata / roc_auc
# ------------------------------------------------------------
# 定義
#   AUC = 「悪性1台と良性1台の組」のうち，悪性のほうがスコアが高い組の割合（同点の組は 0.5 と数える）。
#
# 素直な計算
#   全部の組を1つずつ比べればよい。悪性4台・良性16台なら 4 × 16 = 64 組。
#
# このファイルの計算（結果は同じ。順位を使って一度に求める）
#   AUC = （悪性の順位の合計 − 悪性の台数 ×（悪性の台数 + 1）÷ 2）÷（悪性の台数 × 良性の台数）
#
# なぜこれで求まるか
#   あるクライアントの順位（小さい順）は，「自分以下のスコアの者が，自分を含めて何人いるか」である。
#   悪性1台の順位 = 自分より下にいる良性の数 + 自分より下にいる悪性の数 + 1（自分）。
#   悪性全員ぶんを合計すると，
#       悪性の順位の合計 = 「悪性が良性に勝った組の数」 + 「悪性どうしで付く順位のぶん」
#   になる。後ろの項は，悪性が m 台なら 1 + 2 + ... + m = m(m + 1) / 2 で，スコアによらず決まっている。
#   これを引けば「悪性が良性に勝った組の数」が残り，組の総数で割れば割合（AUC）になる。
#
# 具体例（クライアント5台。悪性は A と C）
#       クライアント   A     B     C     D     E
#       正解          悪性  良性  悪性  良性  良性
#       スコア        0.9   0.2   0.3   0.4   0.1
#   順位（小さい順）: E=1, B=2, C=3, D=4, A=5  ->  悪性の順位の合計 = 5 + 3 = 8
#   悪性どうしのぶん: 2 × 3 ÷ 2 = 3
#   AUC = (8 − 3) ÷ (2 × 3) = 5/6 ≒ 0.833
#   組を数えて確かめると，A は B・D・E の3台すべてに勝ち，C は B・E に勝って D に負けるので，
#   勝ちは 6 組中 5 組。一致する。
#
# 同点の扱い
#   _rankdata は，同じスコアのクライアントに「順位の平均」を付ける（2位と3位が同点なら，どちらも 2.5）。
#   こうすると，同点の組が自動的に 0.5 として数えられる。
#   D1 で「等しいはずのスコアを同じ値にそろえる」処理（_snap_ties）を入れたのは，
#   この仕組みに正しく乗せるためである（norm_deviation_scores のメモを参照）。
#
# ------------------------------------------------------------
# (4) コサイン（cos）で向きを比べること  ->  cosine_distance_scores / structure_diagnostics
# ------------------------------------------------------------
#   2つのベクトル a, b について，  cos = (a と b の内積) ÷ (a の長さ × b の長さ)。
#   長さで割っているので，ベクトルの大きさには左右されず，向きの関係だけが残る。
#       同じ向き    ->  1
#       直角        ->  0
#       正反対      -> -1
#   D2 は 1 − cos をスコアにしている（同じ向きで 0，正反対で 2）。
#   約27万次元のような高次元では，無関係な2つのベクトルはほぼ直角（cos がほぼ 0）になる。
#   そのため，cos がはっきりプラスなら「同じ方向を向いている」，はっきりマイナスなら
#   「逆を向いている」と，意味のある判断ができる。
