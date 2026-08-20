"""
test_malicious.py

malicious.py の単体テスト（実験計画書 7.3 対応）
- malicious_ratio から悪性クライアント数が正しく計算されるか
- seed固定で悪性クライアントidが再現されるか
- 手動指定（malicious_ids文字列）が優先されるか
"""

from malicious import generate_malicious_ids, parse_malicious_ids, resolve_malicious_ids


def check(name, cond, detail=""):
    status = "OK" if cond else "NG"
    print(f"[{status}] {name} {detail}")
    assert cond, f"{name} が失敗しました: {detail}"


def test_ratio_gives_correct_count():
    for ratio, expected_count in [(0.0, 0), (0.1, 10), (0.2, 20), (0.3, 30)]:
        ids = generate_malicious_ids(num_clients=100, malicious_ratio=ratio, seed=42)
        check(
            f"ratio={ratio} -> {expected_count}人",
            len(ids) == expected_count,
            f"actual={len(ids)}",
        )


def test_ids_are_within_range_and_unique():
    ids = generate_malicious_ids(num_clients=100, malicious_ratio=0.3, seed=42)
    check("idに重複がない", len(ids) == len(set(ids)))
    check("idが範囲内(0〜99)", all(0 <= i < 100 for i in ids))


def test_same_seed_is_reproducible():
    ids_a = generate_malicious_ids(num_clients=100, malicious_ratio=0.2, seed=42)
    ids_b = generate_malicious_ids(num_clients=100, malicious_ratio=0.2, seed=42)
    check("同じseedなら同じids", ids_a == ids_b, f"{ids_a} vs {ids_b}")


def test_different_seed_changes_selection():
    ids_a = generate_malicious_ids(num_clients=100, malicious_ratio=0.2, seed=42)
    ids_b = generate_malicious_ids(num_clients=100, malicious_ratio=0.2, seed=1)
    check("seedが違えば選出も変わる（通常は）", ids_a != ids_b, f"{ids_a} vs {ids_b}")


def test_invalid_ratio_raises():
    try:
        generate_malicious_ids(num_clients=100, malicious_ratio=1.5, seed=42)
        ok = False
    except ValueError:
        ok = True
    check("malicious_ratio > 1.0 は例外を出す", ok)

    try:
        generate_malicious_ids(num_clients=100, malicious_ratio=-0.1, seed=42)
        ok = False
    except ValueError:
        ok = True
    check("malicious_ratio < 0.0 は例外を出す", ok)


def test_parse_malicious_ids():
    check("通常のカンマ区切り", parse_malicious_ids("0,1,5", num_clients=10) == [0, 1, 5])
    check("空白を含んでも解釈できる", parse_malicious_ids(" 0, 1 , 5 ", num_clients=10) == [0, 1, 5])
    check("空文字列は空リスト", parse_malicious_ids("", num_clients=10) == [])
    check("空白のみも空リスト", parse_malicious_ids("   ", num_clients=10) == [])

    try:
        parse_malicious_ids("0,1,10", num_clients=10)  # 10は範囲外(0-9のはず)
        ok = False
    except ValueError:
        ok = True
    check("範囲外の値は例外を出す", ok)

    try:
        parse_malicious_ids("0,1,1", num_clients=10)  # 重複
        ok = False
    except ValueError:
        ok = True
    check("重複した値は例外を出す", ok)


def test_resolve_prefers_manual_ids():
    # 手動指定があれば malicious_ratio を無視してそちらを使う
    resolved = resolve_malicious_ids(
        num_clients=100, malicious_ratio=0.3, seed=42, manual_ids_raw="0,1,2"
    )
    check("手動指定が優先される", resolved == [0, 1, 2], f"actual={resolved}")


def test_resolve_falls_back_to_ratio():
    # 手動指定が空なら ratio から自動生成する
    resolved = resolve_malicious_ids(
        num_clients=100, malicious_ratio=0.1, seed=42, manual_ids_raw=""
    )
    expected = generate_malicious_ids(num_clients=100, malicious_ratio=0.1, seed=42)
    check("手動指定が空ならratioから自動生成", resolved == expected, f"actual={resolved}")


if __name__ == "__main__":
    tests = [
        test_ratio_gives_correct_count,
        test_ids_are_within_range_and_unique,
        test_same_seed_is_reproducible,
        test_different_seed_changes_selection,
        test_invalid_ratio_raises,
        test_parse_malicious_ids,
        test_resolve_prefers_manual_ids,
        test_resolve_falls_back_to_ratio,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
