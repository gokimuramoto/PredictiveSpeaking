from predictive_speaking.text import ja


def test_strip_trailing_fillers_basic():
    assert ja.strip_trailing_fillers("このシステムについてえっと") == "このシステムについて"


def test_strip_trailing_fillers_chained_with_punct():
    assert ja.strip_trailing_fillers("それでですね、えっと、あの") == "それでですね、"


def test_strip_trailing_fillers_none():
    assert ja.strip_trailing_fillers("今日はいい天気です") == "今日はいい天気です"


def test_strip_trailing_fillers_all_filler():
    assert ja.strip_trailing_fillers("えっとあの") == ""


def test_dedup_boundary_particle():
    assert ja.dedup_boundary_particle("の進捗状況を", "を報告します") == "報告します"
    assert ja.dedup_boundary_particle("の進捗状況を。", "を報告します") == "報告します"


def test_dedup_boundary_no_false_positive():
    # 助詞でない同一文字は削らない
    assert ja.dedup_boundary_particle("これがわ", "わかりやすい") == "わかりやすい"
    # 異なる助詞は削らない
    assert ja.dedup_boundary_particle("状況を", "に報告") == "に報告"


def test_dedup_boundary_leading_comma():
    assert ja.dedup_boundary_particle("そして", "、次に説明します") == "次に説明します"


def test_natural_cut_at_particle():
    assert ja.natural_cut("この研究の背景には、言語障") == "この研究の背景には、"


def test_natural_cut_no_boundary_keeps_text():
    assert ja.natural_cut("あいうえおかきくけこ") == "あいうえおかきくけこ"


def test_mid_text_filler_kept():
    # 末尾以外のフィラーは触らない(発話済みの内容は書き換えない)
    assert ja.strip_trailing_fillers("えっとこれは大事です") == "えっとこれは大事です"
