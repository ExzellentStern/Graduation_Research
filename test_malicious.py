"""
test_malicious.py

malicious.py の単体テスト（実験計画書 7.3 対応）
- malicious_ratio から悪性クライアント数が正しく計算されるか
- seed固定で悪性クライアントidが再現されるか
- 台数が整数にならない割合がエラーになるか

手動指定（malicious_ids文字列）のテストは unused_features/test_malicious_manual.py にある。
"""

from fl_detection.malicious import generate_malicious_ids


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


def test_non_integer_count_raises():
    # 20台 × 0.125 = 2.5台。整数にならない設定は，丸めずにエラーにする
    try:
        generate_malicious_ids(num_clients=20, malicious_ratio=0.125, seed=1)
        ok = False
    except ValueError:
        ok = True
    check("台数が整数にならない割合は例外を出す", ok)

    # 計算の誤差（100 × 0.3 = 30.000000000000004）は整数として扱う
    check("計算の誤差は整数として扱う", len(generate_malicious_ids(num_clients=100, malicious_ratio=0.3, seed=1)) == 30)
    check("20台・割合0.2 は 4台", len(generate_malicious_ids(num_clients=20, malicious_ratio=0.2, seed=1)) == 4)


if __name__ == "__main__":
    tests = [
        test_ratio_gives_correct_count,
        test_ids_are_within_range_and_unique,
        test_same_seed_is_reproducible,
        test_different_seed_changes_selection,
        test_invalid_ratio_raises,
        test_non_integer_count_raises,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print("\nすべてのテストに合格しました。")
