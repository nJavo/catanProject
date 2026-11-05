from catan.eval.tournament import lineups


def test_rotation_all_order_does_not_depend_on_names():
    """Regression: sorting by spec string mapped rotation r to different seat patterns for different names."""
    a = lineups(["zzz", "champ", "intermediate", "intermediate"], 4, "all")
    b = lineups(["aaa", "champ", "intermediate", "intermediate"], 4, "all")
    assert len(a) == len(b) == 12
    assert [lu.index("zzz") for lu in a] == [lu.index("aaa") for lu in b]
    assert [lu.index("champ") for lu in a] == [lu.index("champ") for lu in b]
