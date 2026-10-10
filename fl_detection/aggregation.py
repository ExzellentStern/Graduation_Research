"""
aggregation.py

集約の実装（実験計画書 7.2 対応）
- モデル重み(state_dict) <-> 1次元ベクトルの相互変換
- FedAvg（重み付き平均）を自作関数として切り出し
- aggregation type（文字列）で切り替える aggregate() ディスパッチ関数

【設計方針】
- 集約の中核ロジックは numpy の2次元配列 (n_clients, dim) を入力とする「純粋関数」として実装する。
  → PyTorch に依存しないため、テストが軽量になり、将来的に別フレームワークでも流用できる。

【現在使える集約手法は fedavg だけ】
検知評価スイープ（sotsuron_plan2.tex 5.2）は集約手法を fedavg に固定しているため，
Robust Aggregation（Median / Trimmed Mean / Krum / Bulyan）は
unused_features/aggregation_robust.py へ移してある（戻し方は unused_features/README.md）。
"""

# 【このファイルの読み方】
#   「集約」の中身を書いたファイル。集約とは，全クライアントから届いたモデルを1つにまとめて，
#   次のラウンドのグローバルモデルを作ること。呼び出しているのは robust_strategy.py の aggregate_train()。
#   attack.py / detection.py と同じく，値を受け取って値を返すだけの関数でできている。
#
#   中身は2種類ある。
#       1) 形の変換 : モデルの重みを「層ごとの辞書」と「1本の長いベクトル」の間で行き来させる。
#                     集約だけでなく，攻撃（ALIE）と検知も，このベクトルの形で計算している。
#       2) 集約     : 全クライアントのベクトルを平均して，1本にまとめる（FedAvg）。
#
#       関数                   robust_strategy.py での使われ方
#       state_dict_to_vector   届いた重み（と，配る前のグローバルモデル）を，ベクトルに並べ直す
#       stack_updates          クライアントごとのベクトルを，1つの表（行 = クライアント）に積む
#       aggregate              表を平均して，1本のベクトルにする（中で fedavg_aggregate を呼ぶ）
#       vector_to_state_dict   平均したベクトルを，層ごとの辞書に戻す
#
# 【なぜ形を変換するのか】
#   モデルの重みは，層ごとに形がばらばらな Tensor の集まりになっている。
#       conv1.weight : 30 × 1 × 3 × 3      conv1.bias : 30
#       conv2.weight : 50 × 30 × 3 × 3     conv2.bias : 50
#       fc1.weight   : 200 × 1250          fc1.bias   : 200
#       fc2.weight   : 10 × 200            fc2.bias   : 10
#   「クライアント間の平均を取る」「更新の大きさや向きを測る」といった計算は，層の区別なく
#   全部の重みを同列に扱いたいので，1列に並べて 266,060 個の数値が並んだ1本のベクトルにする。
#   計算が終わったら，モデルに戻せるように，もとの形に戻す。
#
# 【import の説明】
from collections import OrderedDict                    # 入れた順番を覚えている辞書（Python 標準）。state_dict の形式
from typing import List, Optional, Sequence, Tuple     # 型ヒント用の名前（リスト / 「X か None」/ 順に並んだもの / タプル）

import numpy as np                                     # 数値計算のライブラリ

# torch を読み込めなければ torch = None にして先へ進む（attack.py と同じ書き方）。
# 平均の計算は numpy だけでできるので，torch が無い環境でもそのテストができる。
try:
    import torch
except ImportError:  # このモジュール単体でのテスト時など、torch が無い環境でも import エラーにしない
    torch = None


# ============================================================
# 1) state_dict <-> ベクトル変換
# ============================================================
# 【state_dict_to_vector の説明】
#   層ごとの辞書（state_dict）を，1本のベクトルに並べ直す。
#   返すのは2つ：（ベクトル, 形の情報）。形の情報は，あとで元に戻すときに使う。
#
#   -> Tuple[np.ndarray, List[Tuple[str, tuple]]]
#       返り値の型ヒント。「（numpy 配列, （名前, 形）を並べたリスト）の2つ組を返す」という意味。
#   for k, v in state_dict.items(): : 層を1つずつ順に取り出す（k が層の名前，v がその重みの Tensor）。
#   v.detach().cpu().numpy()        : Tensor を numpy の配列に変える
#                                     （勾配の記録から切り離す -> CPU に置く -> numpy にする）。
#   shapes.append((k, tuple(arr.shape))) : （層の名前, もとの形）を控えておく。例: ("fc2.weight", (10, 200))
#   arr.reshape(-1)                 : 形を崩して1列に並べる。-1 は「長さは自動で決めて」の意味。
#   .astype(np.float64)             : 小数の精度を倍精度（float64）にする。PyTorch の重みは float32 だが，
#                                     平均や検知の計算で誤差がたまりにくいよう，高い精度で計算している。
#   np.concatenate(flat_list)       : 層ごとの1列を，全部つなげて1本にする。
#   A if 条件 else B                : 層が1つも無い場合（空の辞書）は，空の配列を返す。
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


# 【vector_to_state_dict の説明】
#   上の逆。1本のベクトルを，控えておいた形の情報を使って，層ごとの辞書に戻す。
#     vector               : 戻したいベクトル（集約後のもの）
#     shapes               : state_dict_to_vector が返した形の情報
#     reference_state_dict : 見本にする state_dict。各層のデータの種類（float32 など）を合わせるために使う。
#
#   if torch is None: raise ImportError(...)
#       : この関数は Tensor を作るので torch が必要。読み込めていなければ，ここでエラーにする。
#   cursor               : 「ベクトルのここまで使った」位置。0 から始めて，層ごとに進めていく。
#   numel = int(np.prod(shape)) if len(shape) > 0 else 1
#       : その層の重みの個数。np.prod は全部を掛け合わせる（形が (10, 200) なら 10 × 200 = 2000 個）。
#   vector[cursor: cursor + numel] : cursor 番目から numel 個ぶんを切り出す（スライス）。
#   .reshape(shape)      : 切り出した1列を，もとの形に戻す。
#   torch.tensor(chunk, dtype=ref_dtype) : numpy の配列を Tensor にする。種類は見本に合わせる
#                                          （float64 で計算した結果が，ここで float32 に戻る）。
#   最後の if cursor != len(vector):
#       : ベクトルを使い切ったかを確かめる。余ったり足りなかったりしたら，形の情報とベクトルが
#         対応していないということなので，エラーにする（黙って壊れたモデルを作らないため）。
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


# 【stack_updates の説明】
#   クライアントごとのベクトル（20本）を，1つの表に積み重ねる。
#   出来上がる表は，行 = クライアント（20行），列 = 重み1つ1つ（266,060列）。
#     {len(v) for v in vectors} : 集合内包表記。各ベクトルの長さを集めた集合を作る。
#                                 集合は同じ値を1つにまとめるので，全員の長さが同じなら要素は1つになる。
#     if len(lengths) != 1:     : 長さが2種類以上ある（＝モデルの形が違うクライアントがいる）ならエラーにする。
#     np.stack(vectors, axis=0) : ベクトルを縦に積んで，2次元の表にする。
def stack_updates(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """複数クライアントのベクトルを (n_clients, dim) の2次元配列にまとめる。"""
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"全クライアントのベクトル長が一致していません: {lengths}")
    return np.stack(vectors, axis=0)


# ============================================================
# 2) FedAvg（重み付き平均） -- 自作関数として切り出し
# ============================================================
# 【fedavg_aggregate の説明】
#   FedAvg。全クライアントのモデルを，重み付きで平均する。連合学習の一番基本的な集約方法。
#     updates : 全クライアントのベクトルを積んだ表
#     weights : クライアントごとの重み（平均への影響の大きさ）。実験では各クライアントのデータ数を渡している。
#               渡さなかった場合（None）は，全員を同じ重みで平均する。
#   ※ ここでいう「重み」は平均の取り方の重みで，モデルの重み（パラメータ）とは別のもの。
#
#   重み付き平均とは
#     データ数が 100 枚のクライアント A と 300 枚のクライアント B がいて，ある値が A は 2，B は 6 なら，
#         (100 × 2 + 300 × 6) ÷ (100 + 300) = 5
#     単純な平均（4）より，データの多い B の値に近くなる。
#     「データをたくさん持っているクライアントの学習結果ほど，重く扱う」という考え方。
#   この性質のため，データの多い悪性クライアントは影響が大きく，少ない悪性クライアントは影響が小さい。
#   FedAvg には悪性の値を取り除く仕組みが無いので，届いたものはすべて平均に入る。
#
#   n = updates.shape[0]      : 表の行数（クライアントの台数）。
#   np.full(n, 1.0 / n)       : 重みが渡されなかった場合，全員に 1/n を割り当てる。
#   w.shape[0] != n           : 重みの個数がクライアントの台数と合わなければエラーにする。
#   if total <= 0:            : 重みの合計が 0 以下なら割り算ができないので，エラーにする。
#   w = w / total             : 合計が 1 になるように割る（割合に直す）。
#   np.average(updates, axis=0, weights=w)
#       : 列ごと（重み1つ1つについて）に，クライアント間の重み付き平均を取る。結果は1本のベクトル。
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
# 3) ディスパッチ関数（aggregation type で切り替え）
# ============================================================
# 【この部分の説明】
#   集約手法の名前（文字列）を受け取って，その手法の関数を呼ぶ「振り分け役」。
#   pyproject.toml の aggregation-method の値が，ここまで渡ってくる。
#   今は "fedavg" しか使えないので振り分ける先は1つだが，ほかの手法を戻すときは，
#   ここに分岐を足すだけで済む（呼び出す側の robust_strategy.py は変えなくてよい）。
#
#   SUPPORTED_METHODS = ("fedavg",)
#       使える手法の名前の一覧。("fedavg",) の最後のカンマは「要素が1つのタプル」を表す書き方。
#   引数の並びにある * : これより後ろ（weights）は 名前=値 の形で渡す決まり。
#   name = method.lower() : 小文字にそろえる（"FedAvg" と書かれても通るように）。
#   if name == "fedavg": return fedavg_aggregate(updates, weights)
#       : 手法が fedavg なら，FedAvg で集約した結果を返して終わる。
#   そこで終わらなかった（＝知らない名前だった）場合は，raise ValueError(...) でエラーにする。
#   {method!r} は，値を引用符つきで表示する書き方。
SUPPORTED_METHODS = ("fedavg",)


def aggregate(method: str, updates: np.ndarray, *, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    aggregation type（文字列）に応じて集約手法を切り替えるディスパッチ関数。

    Args:
        method: "fedavg"（現在はこれだけ。他の手法は unused_features/aggregation_robust.py）
        updates: shape (n_clients, dim) の各クライアント更新
        weights: データ数による重み付け等
    """
    name = method.lower()
    if name == "fedavg":
        return fedavg_aggregate(updates, weights)
    raise ValueError(
        f"未対応の aggregation method です: {method!r}（対応: {SUPPORTED_METHODS}）。"
        "median / trimmed_mean / krum / bulyan は unused_features/aggregation_robust.py に移してあります"
        "（戻し方は unused_features/README.md）。"
    )
