"""
attack.py

Model Poisoning Attack の実装（sotsuron_plan2.tex 4.3 対応）

- sign_flip_delta（A1：符号反転）:
    「更新量ベース」の単純な攻撃。1クライアントだけで完結するローカルな攻撃で、
    torch state_dict（パラメータ名→Tensor の辞書）を直接操作する関数として実装し、
    client_app.py から呼び出す。
    Δ = local - global を計算 → 攻撃を適用 → global + Δ_malicious に戻す、という
    一連の流れは client_app.py 側で行う（このファイルは攻撃そのものだけ担当）。

- alie_attack（A2：ALIE）:
    正常更新の座標ごとの平均・標準偏差から悪性更新をクラフトするステルス攻撃。
    悪性クライアント同士が結託して他クライアントの更新を参照できることを前提とする。
    Flower のクライアントプロセスは互いに独立しているため，この前提を素直に満たせるのは
    サーバ側だけである。そのためサーバ側（robust_strategy.py の aggregate_train）から，
    「全クライアントの更新を受け取った直後・集約する前」に呼び出し，
    悪性クライアント分のベクトルを攻撃済みベクトルに差し替える。
    numpy の2次元配列 (n_clients, dim) を扱う torch非依存の純粋関数として実装する。

【現在使える攻撃は sign_flip と alie だけ】
Scale 攻撃（scale_delta）と Fang らの Adaptive Model Poisoning Attack（adaptive_attack）は
unused_features/attack_adaptive.py へ移してある（戻し方は unused_features/README.md）。
"""

# 【このファイルの読み方】
#   攻撃の「中身」だけを書いたファイル。ここにある関数は，値を受け取って値を返すだけで，
#   ファイルの読み書きも Flower とのやり取りもしない（こういう関数を「純粋関数」と呼ぶ）。
#   「いつ・誰に対して攻撃するか」を決めているのは呼び出す側で，このファイルではない。
#
#       関数              呼び出す側                              動く条件
#       sign_flip_delta   client_app.py の train()                攻撃が sign_flip で，そのクライアントが悪性のとき
#       alie_attack       robust_strategy.py の aggregate_train() 攻撃が alie で，悪性が1台以上いるとき
#       alie_z_max        テスト（test_attack.py）だけ            実験中は呼ばれない（z の参照値を求める計算）
#
#   2つの攻撃は，扱うデータの形が違う。
#       sign_flip_delta : 1クライアントぶんの更新量。「層の名前 -> Tensor」の辞書（PyTorch の形）。
#       alie_attack     : 全クライアントぶんの重みを並べた表（行 = クライアント，列 = 重み1つ1つ。numpy の形）。
#
# 【import の説明】（import 文の読み方は server_app.py の冒頭を参照）
import math                          # 数学の関数（ここでは切り捨ての math.floor を使う）。Python 標準
from statistics import NormalDist    # 正規分布を扱う道具。Python 標準
from typing import Dict, Sequence    # 型ヒント用の名前。Dict は辞書，Sequence は「順に並んだもの」（リストや配列）

import numpy as np                   # 数値計算のライブラリ

# try: ... except ImportError: ...
#   torch を読み込んでみて，読み込めなければ（インストールされていなければ）エラーにせず torch = None にする。
#   このファイルで torch を使うのは型ヒントの中だけなので，torch が無い環境でも ALIE のテストができる。
try:
    import torch
except ImportError:  # torch が無い環境でも import エラーにしない（テスト用）
    torch = None


# ============================================================
# 1) sign_flip （更新量ベース、クライアント側で使用）
# ============================================================
# 【この部分の説明】
#   符号反転攻撃。更新量 Δ（学習でモデルがどれだけ動いたか）の向きを，正反対にする。
#   正直なクライアントは「損失が下がる向き」にモデルを動かすので，その逆は「損失が上がる向き」になる。
#   全体の平均に混ざると，学習を逆方向へ引っ張る。
#
#   向きが正反対になるだけで，動かす大きさは変わらない。
#   そのため，更新の大きさを見る検知器 D1 では見分けにくく，向きを見る検知器 D2 では見分けやすい。
#
#   def sign_flip_delta(delta: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
#       delta : 更新量。「層の名前 -> その層の重みの変化（Tensor）」の辞書。
#       Dict[str, "torch.Tensor"] は「名前（文字列）-> Tensor の辞書」という型ヒント。
#       型の名前を " " で囲んでいるのは，torch が読み込めなかった場合でもエラーにしないため。
#   {k: -v for k, v in delta.items()}
#       辞書内包表記。辞書から（名前 k, 値 v）を1組ずつ取り出し，値にマイナスを付けた新しい辞書を作る。
#       -v は Tensor の全要素の符号を反転する（1.5 は -1.5 に，-0.2 は 0.2 に）。
#       新しい辞書を作って返すので，渡された delta そのものは書き換えない。
def sign_flip_delta(delta: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Δ の符号を反転する（Sign Flipping Attack）。"""
    return {k: -v for k, v in delta.items()}


# ============================================================
# 2) ALIE （サーバ側で使用）
# ============================================================
# 【alie_z_max の説明】
#   原論文の式に当てはめて，「z をどこまで大きくしても見つからずに済むか」の目安を計算する関数。
#   実験中は呼ばれない。pyproject.toml の alie-zs にある 0.385 という値の根拠を，テストで確かめるためにある。
#
#   def alie_z_max(n: int, f: int) -> float:
#       n : クライアントの総数，f : 悪性クライアントの数。小数（float）を返す。
#   s = math.floor(n / 2 + 1) - f
#       math.floor は小数点以下の切り捨て。n=20, f=4 なら floor(11) - 4 = 7。
#   if not 0 < s < n:
#       0 < s < n は「s が 0 より大きく，n より小さい」をまとめて書いたもの。not でそれを否定している。
#       つまり s がこの範囲から外れていたら（悪性が過半数など，式が成り立たない場合）エラーにする。
#   raise ValueError(...)
#       エラーを起こして処理を止める。
#   NormalDist().inv_cdf(p)
#       標準正規分布（平均0・標準偏差1）で，「それ以下になる確率がちょうど p になる値」を返す。
#       n=20, f=4 なら p = (20 - 7) / 20 = 0.65 で，返る値は約 0.385。
def alie_z_max(n: int, f: int) -> float:
    """
    Baruch et al. (2019) Algorithm 3 の z_max を返す（参照値）。

        s     = floor(n/2 + 1) - f
        z_max = Phi^{-1}((n - s) / n)        （Phi は標準正規の累積分布関数）

    「悪性値より外側に正常 worker が s 個いれば Median 等に紛れ込める」という
    ランク条件を，正常更新の各座標が正規分布に従うという仮定の下で z に換算したもの。
    n=20, f=4 なら s=7, z_max = Phi^{-1}(0.65) ≒ 0.385。

    注意：これは原論文の分布仮定（正常 worker が IID かつ座標ごとに正規分布）の下での
    参照値であり，Non-IID 環境で同じランク上の意味を持つことは保証されない。
    """
    s = math.floor(n / 2 + 1) - f
    if not 0 < s < n:
        raise ValueError(f"z_max が定義できません（n={n}, f={f}, s={s}）")
    return NormalDist().inv_cdf((n - s) / n)


# 【alie_attack の説明】
#   ALIE 攻撃。良性クライアントの集団の「ばらつきの範囲内」に収まるような偽の重みを作り，
#   悪性クライアント全員に同じものを持たせる。極端な値を送らないので，見つかりにくい。
#
#   考え方（重み1つ1つについて，別々に同じことをする）
#       1. 良性クライアントの値の平均 mu と，ばらつきの大きさ（標準偏差）sigma を求める。
#       2. 平均から，標準偏差の z 倍だけずらした値  mu - z * sigma  を作る。
#       3. それを悪性クライアントの値にする。
#     例: ある重みについて，良性の値の平均が 0.50，標準偏差が 0.10，z = 1.0 なら，悪性の値は 0.40。
#         良性の値は 0.40〜0.60 あたりに散らばっているので，0.40 は「端のほうにいる良性」に見える。
#   z が大きいほど平均から遠ざかり，攻撃としては強くなるが，目立ちやすくなる。
#   z が小さいほど良性に紛れるが，モデルを動かす力は弱くなる。この兼ね合いを z の値で調べている。
#
#   引数
#       vectors       : 全クライアントの重みを並べた表（行 = クライアント，列 = 重み1つ1つ）。
#       malicious_idx : 悪性クライアントが，表の何行目にいるかの一覧。
#       benign_idx    : 良性クライアントが，表の何行目にいるかの一覧。
#       z             : ずらす強さ。pyproject.toml の値がここまで渡ってくる。
#   返すもの          : 悪性の行だけを差し替えた，新しい表。
#
#   ※ 渡される表の中身は「更新量」ではなく「重みそのもの」（配られた重み + 更新量）だが，結果は同じになる。
#     全員が同じ重みから出発しているので，平均は「配られた重み + 更新量の平均」，
#     標準偏差は「更新量の標準偏差」と同じになり，mu - z * sigma は
#     「配られた重み +（更新量の平均 − z × 更新量の標準偏差）」になるため。
def alie_attack(
    vectors: np.ndarray,
    malicious_idx: Sequence[int],
    benign_idx: Sequence[int],
    z: float,
) -> np.ndarray:
    """
    ALIE（Baruch et al., "A Little Is Enough", NeurIPS 2019）に基づく悪性更新生成則。
    研究計画書（改訂版, sotsuron_plan2.tex）の A2。

    正常クライアントの更新の「座標ごとの平均 mu・標準偏差 sigma」を求め，
    悪性クライアント全員に同一の

        g_mal = mu - z * sigma

    を割り当てる。

    【原論文との関係】原論文 Algorithm 3 をそのまま再現したものではない。
    - z は原論文の z_max（alie_z_max 参照）から導出せず，実験条件として与えた値を
      そのまま使う（pyproject.toml の alie-z / [tool.sweep] の alie-zs。この関数に既定値は置かない）。
      参考：n=20, f=4 の z_max は約 0.385。
    - 原論文は mu + z_max * sigma の形で提示している（正規分布は対称なので
      「平均から標準偏差方向へずらす」という発想は同じ）。
    - mu, sigma は当該ラウンドの正常更新そのものから計算する（full-knowledge）。
    - 原論文の摂動範囲の議論は正常 worker の分布仮定に依存する。Non-IID 環境で
      同じ理論的意味を持つとは仮定せず，同一の生成則を異なる Non-IID 条件へ
      適用する実験的な攻撃モデルとして使う。

    Args:
        vectors: shape (n_clients, dim)。各クライアントの（攻撃前の）モデルベクトル。
        malicious_idx: 悪性クライアントのインデックス（vectors内での位置）。
        benign_idx: 正常クライアントのインデックス（vectors内での位置）。
        z: 摂動の強さ（標準偏差の何倍だけ平均からずらすか）。
    """
    # --- 受け取った値の形をそろえる ---
    #   np.asarray(x, dtype=...) : x を numpy の配列にする。dtype は中身の種類（float64 は小数，int は整数）。
    #   set(int(i) for i in malicious_idx) : 番号を整数にして集合に入れる（同じ番号が2回あっても1つになる）。
    #   sorted(...)                        : 小さい順に並べたリストにする。
    vectors = np.asarray(vectors, dtype=np.float64)
    malicious_idx = np.asarray(sorted(set(int(i) for i in malicious_idx)), dtype=int)
    benign_idx = np.asarray(benign_idx, dtype=int)

    # --- 攻撃できない場合は，何も変えずに返す ---
    #   A or B          : どちらか一方でも成り立てば実行する（悪性が0台，または良性が0台）。
    #   vectors.copy()  : 表の写しを作る。もとの表をそのまま返さないのは，
    #                     呼び出した側が返り値を書き換えても，もとの表に影響しないようにするため。
    if len(malicious_idx) == 0 or len(benign_idx) == 0:
        return vectors.copy()

    # --- 良性の平均と標準偏差を求める ---
    #   vectors[benign_idx] : 表から良性の行だけを取り出す。
    #   .mean(axis=0)       : 列ごと（＝重み1つ1つについて）に，良性クライアント間の平均を取る。
    #   .std(axis=0)        : 同じく列ごとに，標準偏差（ばらつきの大きさ）を取る。
    #   mu と sigma は，どちらも重みの数（約27万）だけ値が並んだ1本のベクトルになる。
    benign = vectors[benign_idx]
    mu = benign.mean(axis=0)
    sigma = benign.std(axis=0)  # 母標準偏差（ddof=0）

    # --- 偽の重みを作り，悪性の行に入れる ---
    #   payload = mu - z * sigma : 重み1つ1つについて「平均 − z × 標準偏差」を計算する。
    #                              ベクトルどうしの計算は，同じ位置の値どうしで行われる。
    #   crafted = vectors.copy() : 表の写しを作り，こちらを書き換える（もとの表は変えない）。
    #   crafted[idx] = payload   : idx 行目を，まるごと payload に置き換える。
    #   悪性クライアントは全員まったく同じ行になる。良性の行は変わらない。
    crafted = vectors.copy()
    payload = mu - z * sigma
    for idx in malicious_idx:
        crafted[idx] = payload
    return crafted

