from predictive_speaking.score import leading_match_len, percentile


def test_leading_match_basic():
    assert leading_match_len("リアルタイムに解析", "リアルタイムに解析して表示") == 8


def test_leading_match_punct_ignored():
    assert leading_match_len("解析して、表示", "解析して表示します") == 6


def test_leading_match_mismatch():
    assert leading_match_len("全然違う内容", "リアルタイムに") == 0


def test_leading_match_cap():
    assert leading_match_len("あいうえおかきくけこ", "あいうえおかきくけこ", cap=5) == 5


def test_percentile():
    vals = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(vals, 50) == 3.0
    assert percentile(vals, 95) == 100.0
