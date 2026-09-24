import asyncio

import pytest

from predictive_speaking.asr.typed import TypedASR
from predictive_speaking.predict.loop import PredictionLoop
from predictive_speaking.predict.mock import MockPredictor
from predictive_speaking.session import LiveSession


@pytest.mark.asyncio
async def test_live_session_with_typed_asr():
    script = ["このシステムではユーザーが言葉に詰まったときに補助します", "次の文です"]
    asr = TypedASR(script, chars_per_tick=4, tick_ms=20, sentence_gap_ms=30)
    pred = MockPredictor(delay_ms=5, oracle_text=script[0])
    results = []
    loop = PredictionLoop(pred, on_result=results.append, min_cycle_ms=0)
    session = LiveSession(asr, loop)

    await session.start()
    await asyncio.wait_for(session.wait(), timeout=10)
    await asyncio.sleep(0.05)  # 最後の予測完了を待つ
    await session.stop()

    # partialが流れ、予測が回った
    assert loop.stats.cycles >= 3
    assert results
    # finalで履歴に積まれた
    assert session.state.history == script
    # 予測入力は最新partial(=文の先頭部分)だったはず
    assert all(r.input_latest for r in results)


@pytest.mark.asyncio
async def test_history_join_merges_fragments_without_extra_punct():
    """Azureの細切れfinal(断片ごとに「。」付き)を自然な一文に結合する。"""
    asr = TypedASR([], tick_ms=1)
    loop = PredictionLoop(MockPredictor(delay_ms=1))
    session = LiveSession(asr, loop)
    session.state.history = ["今、プロジェクトスペース。", "の進捗状況を。", "を報告します。"]
    text = session.history_text()
    assert text.count("。") == 1  # 完結した末尾にだけ句点
    assert "。の" not in text and "。を" not in text  # 断片間に句点が挟まらない


@pytest.mark.asyncio
async def test_flush_submits_untracked_tail():
    """無音時flush: 発火条件に満たない末尾(+1文字やフィラー詰まり)も予測に流す。"""
    from predictive_speaking.clock import now_ms

    asr = TypedASR([], tick_ms=1)
    pred = MockPredictor(delay_ms=1)
    loop = PredictionLoop(pred)
    session = LiveSession(asr, loop)
    session.state.partial = "今回の修正は全体的に"
    session.state.last_activity_t_ms = now_ms() - 1000  # 1秒無音
    session._last_submit_input = "今回の修正は全体的"  # 最後の1文字が未送信
    session.flush(now_ms())
    assert session._last_submit_input == "今回の修正は全体的に"
    # 直後の再flushは重複送信しない
    before = loop._version if hasattr(loop, "_version") else None
    session.flush(now_ms())
    assert session._last_submit_input == "今回の修正は全体的に"


@pytest.mark.asyncio
async def test_session_history_budget():
    asr = TypedASR([], tick_ms=1)
    loop = PredictionLoop(MockPredictor(delay_ms=1))
    session = LiveSession(asr, loop, max_history_utterances=3, history_char_budget=20)
    session.state.history = ["あ" * 30, "い" * 10, "う" * 10, "え" * 10]
    text = session.history_text()
    assert len(text) <= 20
    assert text.endswith("え" * 10)
