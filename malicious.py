"""
malicious.py

悪性クライアントの選出（実験計画書 7.3 対応）
- malicious_ratio から悪性クライアント数を計算し、seed固定で悪性クライアントidを生成する
- 手動指定（malicious_ids文字列 / seed）にも対応し、指定があればそちらを優先する
  （seed・malicious_idsを個別に把握・操作したいという実験上の要求に対応するため）

【設計方針】
- ratioからの自動生成:
    num_malicious = round(num_clients * malicious_ratio)
    np.random.default_rng(seed) で num_clients から num_malicious 個を重複無しで抽出する
  → 同じ (num_clients, malicious_ratio, seed) の組なら常に同じ集合になる（再現性）
- 手動指定:
    "0,1,5" のようなカンマ区切り文字列を int のリストへパースする
    範囲外の値・重複があれば ValueError
- resolve_malicious_ids() が両者の入口。manual_ids_raw が空でなければそちらを優先する。
"""

from typing import List

import numpy as np


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

    num_malicious = round(num_clients * malicious_ratio)
    rng = np.random.default_rng(seed)
    ids = rng.choice(num_clients, size=num_malicious, replace=False)
    return sorted(int(i) for i in ids)


def parse_malicious_ids(raw: str, num_clients: int) -> List[int]:
    """
    "0,1,5" のようなカンマ区切り文字列を手動指定のidリストへ変換する。
    空文字列（または空白のみ）の場合は空リストを返す（=手動指定なし、の意味）。
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    ids = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if not (0 <= idx < num_clients):
            raise ValueError(
                f"malicious_ids に範囲外の値があります: {idx}（num_clients={num_clients}）"
            )
        ids.append(idx)

    if len(ids) != len(set(ids)):
        raise ValueError(f"malicious_ids に重複があります: {ids}")

    return sorted(ids)


def resolve_malicious_ids(
    num_clients: int,
    malicious_ratio: float,
    seed: int,
    manual_ids_raw: str = "",
) -> List[int]:
    """
    悪性クライアントidの決定の入口。
    manual_ids_raw（手動指定）が空でなければそちらを優先し、
    空なら malicious_ratio + seed から自動生成する。
    """
    manual_ids = parse_malicious_ids(manual_ids_raw, num_clients)
    if manual_ids:
        return manual_ids
    return generate_malicious_ids(num_clients, malicious_ratio, seed)
