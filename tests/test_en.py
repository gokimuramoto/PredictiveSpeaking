"""英語モード(text/en + validate lang="en" + LiveSession en)の単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.text import en
from predictive_speaking.validate import validate


# ---------- text/en ----------

def test_en_normalize_keeps_spaces():
    assert en.normalize("Hello   world ") == "Hello world"


def test_en_normalize_for_match_strips_punct_and_case():
    assert en.normalize_for_match("Hello, World!") == en.normalize_for_match("hello world")


def test_en_is_incomplete():
    assert en.is_incomplete("I think we should")
    assert en.is_incomplete("Let me check the")
    assert not en.is_incomplete("That works for me.")


def test_en_strip_trailing_fillers():
    assert en.strip_trailing_fillers("we can do that um") == "we can do that"
    assert en.strip_trailing_fillers("so the plan is, you know") == "so the plan is"
    assert en.strip_trailing_fillers("um") == ""


def test_en_strip_leading_fillers():
    assert en.strip_leading_fillers("um, I think so") == "I think so"
    assert en.strip_leading_fillers("you know, right") == "right"


def test_en_natural_cut_word_boundary():
    # 途中で切れた単語を落とし、続けて末尾に残る冠詞・接続語も落とす
    assert en.natural_cut("we should finish the repor") == "we should finish"
    assert en.natural_cut("see you tomorrow.") == "see you tomorrow."


def test_en_fit_length():
    long = "this is a very long sentence that keeps going and going beyond the limit"
    cut = en.fit_length(long, 40)
    assert len(cut) <= 40
    assert not cut.endswith(" ")


# ---------- validate(lang="en") ----------

def test_validate_en_accepts_continuation():
    v = validate(" and then we can review it together.", "I will finish the draft", lang="en")
    assert v.speakable, v.flags
    assert v.text.startswith("and then")


def test_validate_en_default_max_chars_is_60():
    # lang="en"でmax_chars未指定なら60が適用される(ja既定の22で切られない)
    v = validate(" and then we can review it together tomorrow", "I will finish", lang="en")
    assert v.speakable, v.flags
    assert len(v.text) > 22


def test_validate_en_rejects_degenerate():
    v = validate(" yes yes yes yes yes yes", "I think", lang="en")
    assert not v.speakable
    assert "degenerate" in v.flags


def test_validate_en_rejects_word_level_degenerate():
    # 前置き付きの反復(「was the the the...」)は文字単位検出をすり抜けるため単語レベルで弾く
    v = validate(" was the the the the the and the the the", "people he", lang="en")
    assert not v.speakable
    assert "degenerate" in v.flags


def test_validate_en_allows_normal_the_usage():
    v = validate(" and the main point is the latency of the system.", "I think", lang="en")
    assert v.speakable, v.flags


def test_validate_en_cuts_overlong():
    raw = (" and we could also invite the design team and the marketing team"
           " and everyone else who might care")
    v = validate(raw, "next week", max_chars=60, lang="en")
    if v.speakable:
        assert len(v.text) <= 60
        assert "truncated" in v.flags


def test_validate_en_rejects_assistant_meta():
    v = validate(" As an AI assistant, I cannot do that", "I said", lang="en")
    assert not v.speakable


# ---------- LiveSession en ----------

def test_session_history_text_en_joins_with_spaces():
    from predictive_speaking.session import LiveSession

    class _DummyASR:
        async def start(self):  # pragma: no cover
            pass

        async def stop(self):  # pragma: no cover
            pass

        def events(self):  # pragma: no cover
            raise NotImplementedError

    class _DummyLoop:
        def submit(self, *a, **k):  # pragma: no cover
            pass

        async def run(self):  # pragma: no cover
            pass

        def stop(self):  # pragma: no cover
            pass

    s = LiveSession(_DummyASR(), _DummyLoop(), lang="en")
    # AzureのEN finalは終止符付きで来る。断片(無終止符)は句点を付けずに連結される
    s.state.history = ["I sent the report.", "and the follow-up", "is scheduled."]
    text = s.history_text()
    assert text == "I sent the report. and the follow-up is scheduled."
