from predictive_speaking.validate import SILENCE_TOKEN, agreement_len, validate


def test_normal_continuation():
    v = validate("次の言葉を予測して", "このシステムではユーザーが言葉に詰まったとき")
    assert v.speakable
    assert v.text == "次の言葉を予測して"
    assert v.flags == []


def test_silence_token():
    v = validate(SILENCE_TOKEN, "本日はありがとうございました")
    assert not v.speakable
    assert v.silence


def test_assistant_like_rejected():
    v = validate("こちらこそよろしくお願いします", "こんにちは")
    assert not v.speakable
    assert "assistant_like" in v.flags


def test_question_flagged_but_speakable():
    v = validate("どう思いますか？", "この案について")
    assert "question" in v.flags


def test_digits_flagged():
    v = validate("三十パーセントは30です", "割合は")
    assert "digits" in v.flags


def test_echo_rejected():
    v = validate("詰まったとき", "このシステムではユーザーが言葉に詰まったとき")
    assert "echo" in v.flags
    assert not v.speakable


def test_truncation():
    v = validate("これはとても長い候補でありましてどう考えても上限の二十二文字を超えています", "テスト")
    assert "truncated" in v.flags
    assert len(v.text) <= 22


def test_meta_rejected():
    v = validate("出力: 続きです", "テスト")
    assert not v.speakable


def test_first_sentence_only():
    v = validate("解析して。次に生成します", "音声を")
    assert v.text == "解析して"


def test_degenerate_low_diversity_variant():
    """「ああああらああぁあい」型(単純反復でない退化)も検出する。"""
    v = validate("あああああらああああああぁあああああいあああ", "今から")
    assert not v.speakable
    assert "degenerate" in v.flags


def test_leading_filler_stripped_from_candidate():
    v = validate("えっと、発話予測のテストをします", "今から")
    assert v.text == "発話予測のテストをします"


def test_no_next_sentence_jump_when_partial_incomplete():
    """発話が未完結の途中では、次の文へ飛ばず短い断片を保持する。"""
    v = validate("は。次の議題ですが", "その点について")
    assert "next_sentence" not in v.flags


def test_next_sentence_when_first_is_tail_remnant():
    """「〜です」への「ね」だけの補完は避け、次の文の内容を候補にする。"""
    v = validate("ね。それでは次に説明します", "これで実装は完成です")
    assert v.text == "それでは次に説明します"
    assert "next_sentence" in v.flags
    assert v.speakable


def test_trivial_tail_rejected_when_sentence_complete():
    v = validate("ね", "これで実装は完成です")
    assert not v.speakable
    assert "trivial_tail" in v.flags


def test_short_completion_ok_when_incomplete():
    """発話が未完結なら短い補完(語の続き)は正当。"""
    v = validate("ればいけない", "この点は考えなけ")
    assert v.speakable


def test_degenerate_internal_repetition():
    v = validate("ああああああああああ", "ああ")
    assert not v.speakable
    assert "degenerate" in v.flags


def test_degenerate_tail_echo_repetition():
    v = validate("、マイクテスト、マイクテスト、", "マイクテスト")
    assert not v.speakable
    assert "degenerate" in v.flags


def test_degenerate_comma_loop():
    v = validate("、の、の、の、の、の", "の")
    assert not v.speakable


def test_normal_not_degenerate():
    v = validate("を報告します", "本日の進捗状況")
    assert v.speakable


def test_particle_dedup_at_junction():
    """発話末尾と候補先頭の助詞重複は縫合される(「をを」防止)。"""
    v = validate("を報告します", "本日の進捗状況を")
    assert v.text == "報告します"
    assert "particle_dedup" in v.flags
    assert v.speakable


def test_agreement():
    assert agreement_len(["リアルタイムに", "リアルタイムで", "リアルな"]) == 3
    assert agreement_len(["予測して"]) == 4
    assert agreement_len([]) == 0
