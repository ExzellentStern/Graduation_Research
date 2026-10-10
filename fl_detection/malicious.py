"""
malicious.py

悪性クライアントの選出（実験計画書 7.3 対応）
- malicious_ratio から悪性クライアント数を計算し、seed固定で悪性クライアントidを生成する

【設計方針】
- num_malicious = num_clients * malicious_ratio（整数にならない設定はエラーにする）
- np.random.default_rng(seed) で num_clients から num_malicious 個を重複無しで抽出する
  → 同じ (num_clients, malicious_ratio, seed) の組なら常に同じ集合になる（再現性）

【現在は使っていない機能】
悪性クライアントの番号を文字列（"0,1,5" など）で手動指定する機能は，
unused_features/malicious_manual.py へ移してある（戻し方は unused_features/README.md）。
"""

# 【このファイルの読み方】
#   「どのクライアントを悪性にするか」を決めるファイル。実験の最初に1回だけ使われる。
#   ここで決まった番号の一覧が，戦略（RobustFedAvg）と全クライアントに配られ，実験の最後まで変わらない。
#
#   ここで決めるのは「誰が悪性か」だけ。「どう攻撃するか」は attack.py，
#   「攻撃をいつ適用するか」は client_app.py と robust_strategy.py の役割。
#
#   呼び出しているのは server_app.py の main() で，generate_malicious_ids() を呼んでいる。
#   関数はこの1つだけ。台数・割合・乱数の種から，悪性にするクライアントを選ぶ。
#
#   クライアントの番号は 0 から始まる。20台なら 0〜19 番。
#
# 【import の説明】
from typing import List  # 型ヒント用の名前。List[int] は「整数のリスト」

import numpy as np       # 数値計算のライブラリ（乱数で選ぶのに使う）


# 【generate_malicious_ids の説明】
#   悪性クライアントを選ぶ。
#     1. 割合が 0〜1 の範囲か，台数が正の数かを確かめる（おかしな設定なら，ここで止める）。
#     2. 悪性の台数を決める：台数 × 割合。これが整数にならない設定は，エラーにして止める。
#     3. 0 から（台数 − 1）までの番号の中から，その台数ぶんを，重複なしで乱数で選ぶ。
#     4. 小さい順に並べて返す。
#   例: 20台・割合 0.2 なら 4台。乱数の種が 1 のとき [8, 9, 14, 19]，2 のとき [2, 4, 5, 14]，3 のとき [1, 3, 4, 13]。
#
#   乱数の種（seed）
#     同じ種なら，何度実行しても同じ番号が選ばれる。違う種なら，別の番号が選ばれる。
#     スイープでは種に trial 番号を渡しているので，trial ごとに悪性になるクライアントが変わる。
#     「たまたまどのクライアントが悪性だったか」に結果が左右されないよう，複数の trial で平均を取るため。
#
#   if not (0.0 <= malicious_ratio <= 1.0):
#       0.0 <= x <= 1.0 は「x が 0 以上 1 以下」をまとめて書いたもの。not でそれを否定している。
#   exact = num_clients * malicious_ratio : 悪性の台数（小数のまま）。
#   num_malicious = round(exact)          : 一番近い整数。
#   if abs(exact - num_malicious) > 1e-9: : 2つの差が，計算の誤差では説明できないほど大きければ
#                                           （＝もともと整数でなければ）エラーにする。
#       abs は絶対値。1e-9 は 0.000000001。
#       100 × 0.3 がコンピュータの計算では 30.000000000000004 になる，といった誤差は許し，
#       20 × 0.125 = 2.5 のような，本当に整数でないものだけを止める。
#   rng = np.random.default_rng(seed)       : seed を種にした乱数の発生器を作る。
#   rng.choice(num_clients, size=台数, replace=False)
#       0 〜 num_clients − 1 の中から，size 個を選ぶ。replace=False は「同じ番号を2回選ばない」。
#   sorted(int(i) for i in ids)             : numpy の整数を普通の整数に直し，小さい順に並べたリストにする。
def generate_malicious_ids(num_clients: int, malicious_ratio: float, seed: int) -> List[int]:
    """
    malicious_ratio・num_clients・seed から悪性クライアントidを自動生成する。

    Args:
        num_clients: クライアント総数
        malicious_ratio: 悪性クライアントの割合（0.0〜1.0）
        seed: 再現性用シード
    """
    if not (0.0 <= malicious_ratio <= 1.0):
        raise ValueError(f"malicious_ratio は 0.0〜1.0 の範囲で指定してください: {malicious_ratio}")
    if num_clients <= 0:
        raise ValueError(f"num_clients は正の整数である必要があります: {num_clients}")

    # 【メモ：整数にならない設定をエラーにしていることについて（2026-10-09）】
    # 以前は round() で丸めていた。Python の round は，ちょうど .5 のとき偶数のほうへ丸める
    # （round(2.5) は 3 ではなく 2）ので，20台・割合 0.125 のような設定だと，黙って 2台になっていた。
    # その場合，ログには「割合 0.125」と記録されるのに，実際は 2 ÷ 20 = 0.1 で，記録と実態がずれる。
    # 割合は実験条件を表す値なので，台数にぴったり対応する値しか受け付けないことにした。
    exact = num_clients * malicious_ratio
    num_malicious = round(exact)
    if abs(exact - num_malicious) > 1e-9:
        raise ValueError(
            f"{num_clients} 台 × 割合 {malicious_ratio} = {exact} 台で，整数になりません。"
            "悪性クライアントの台数が整数になる割合を指定してください"
            f"（例: {num_malicious} 台にするなら {num_malicious / num_clients}）。"
        )
    rng = np.random.default_rng(seed)
    ids = rng.choice(num_clients, size=num_malicious, replace=False)
    return sorted(int(i) for i in ids)
