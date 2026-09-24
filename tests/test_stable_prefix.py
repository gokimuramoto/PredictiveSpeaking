from predictive_speaking.text.stable_prefix import StablePrefixTracker


def test_growth_and_stable():
    tr = StablePrefixTracker(history_size=3, min_growth_chars=2, max_interval_ms=400)
    u1 = tr.update("このシステムでは", t_ms=0)
    assert u1.should_predict  # 初回は発火
    assert u1.stable == "このシステムでは"

    u2 = tr.update("このシステムではユーザーが", t_ms=100)
    assert u2.stable == "このシステムでは"  # LCP
    assert u2.tail == "ユーザーが"
    assert u2.should_predict  # 5文字成長

    u3 = tr.update("このシステムではユーザーが", t_ms=150)
    assert not u3.should_predict  # 成長なし


def test_rewind_detection():
    tr = StablePrefixTracker(history_size=2)
    tr.update("今回の研究では音声を", t_ms=0)
    tr.update("今回の研究では音声認識を", t_ms=100)
    u = tr.update("今回の研究とは違って", t_ms=200)
    assert u.rewound
    assert u.should_predict
    assert u.stable == "今回の研究"


def test_interval_fire():
    tr = StablePrefixTracker(min_growth_chars=5, max_interval_ms=300)
    tr.update("こんにちは", t_ms=0)
    u2 = tr.update("こんにちは今日", t_ms=100)
    assert not u2.should_predict  # 2文字成長 < 5
    u3 = tr.update("こんにちは今日は", t_ms=500)
    assert u3.should_predict  # 300ms経過 + 1文字以上成長
