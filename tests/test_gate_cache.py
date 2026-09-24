import asyncio

import pytest

from predictive_speaking.gate import Gate, GateSignals
from predictive_speaking.predict.types import Candidate, CandidateSet
from predictive_speaking.tts.cache import SpeechPrefetcher


class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.calls: list[str] = []

    async def synthesize(self, text: str, language: str = "Japanese"):
        self.calls.append(text)
        await asyncio.sleep(0.01)
        return b"\x00\x01" * 2400, 50.0  # 100ms分のPCM


def make_result(text: str, input_latest: str) -> CandidateSet:
    cand = Candidate(text=text, raw_text=text, source="greedy", speakable=True)
    return CandidateSet(request_id=1, input_latest=input_latest, created_t_ms=0.0,
                        candidates=[cand], confidence=0.8)


@pytest.mark.asyncio
async def test_prefetch_dedupe_and_fresh():
    tts = FakeTTS()
    pf = SpeechPrefetcher(tts)
    pf.on_result(make_result("続きです", "こんにちは今日"))
    await asyncio.sleep(0.05)
    assert pf.current and pf.current.text == "続きです"
    # 同一テキストは再合成しない
    pf.on_result(make_result("続きです", "こんにちは今日は"))
    await asyncio.sleep(0.05)
    assert tts.calls == ["続きです"]
    # 由来partialが現在partialと整合すればfresh
    assert pf.fresh("こんにちは今日は") is not None
    # 全く別の発話に変わったらfreshではない
    assert pf.fresh("全然別の話") is None
    # 発話中でない(partial空)ならfreshではない
    assert pf.fresh("") is None
    # 発話確定で無効化される
    pf.invalidate()
    assert pf.current is None
    assert pf.fresh("こんにちは今日は") is None


@pytest.mark.asyncio
async def test_completed_synth_cached_even_if_desired_moved_on():
    """合成中に新候補が来ても、完成した合成は破棄せずキャッシュする(ライブロック防止)。"""
    class GatedTTS(FakeTTS):
        """テストが明示的に完了させる合成(決定的タイミング)。"""

        def __init__(self):
            super().__init__()
            self.gate = asyncio.Semaphore(0)

        async def synthesize(self, text, language="Japanese"):
            self.calls.append(text)
            await self.gate.acquire()
            return b"\x00\x01" * 2400, 50.0

    tts = GatedTTS()
    pf = SpeechPrefetcher(tts)
    pf.on_result(make_result("候補あああ", "入力1"))
    await asyncio.sleep(0)  # workerが合成Aを開始
    pf.on_result(make_result("候補いいい", "入力2"))  # 合成A中に希望がBへ変わる
    tts.gate.release()  # 合成Aを完了させる
    await asyncio.sleep(0.01)
    # 旧実装はここでAを破棄していた(→キャッシュが埋まらないライブロック)
    assert pf.current is not None and pf.current.text == "候補あああ"
    tts.gate.release()  # 続けて取り直された合成Bを完了させる
    await asyncio.sleep(0.01)
    assert pf.current.text == "候補いいい"
    assert tts.calls == ["候補あああ", "候補いいい"]


@pytest.mark.asyncio
async def test_prefetch_replaces_on_new_text():
    tts = FakeTTS()
    pf = SpeechPrefetcher(tts)
    pf.on_result(make_result("候補A", "入力1"))
    await asyncio.sleep(0.05)
    pf.on_result(make_result("候補B", "入力2"))
    await asyncio.sleep(0.05)
    assert pf.current.text == "候補B"


@pytest.mark.asyncio
async def test_pin_overrides_and_blocks_new_desires():
    tts = FakeTTS()
    pf = SpeechPrefetcher(tts)
    pf.request_pin("ピンした候補です", "入力元")
    pf.on_result(make_result("別の新候補", "入力X"))  # ピン中は無視される
    await asyncio.sleep(0.05)
    assert pf.current is not None and pf.current.text == "ピンした候補です"
    assert "別の新候補" not in tts.calls
    pf.clear_pin()
    pf.on_result(make_result("別の新候補", "入力X"))  # 解除後は通常動作
    await asyncio.sleep(0.05)
    assert pf.current.text == "別の新候補"


def test_gate_button_policy_and_cooldown():
    gate = Gate(cooldown_ms=10000)
    s = GateSignals(button_pressed=True, cache_available=True, playing=False)
    assert gate.should_fire(s)
    # cooldown中は発火しない
    s2 = GateSignals(button_pressed=True, cache_available=True, playing=False)
    assert not gate.should_fire(s2)
    # ボタンなし・キャッシュなしは発火しない
    gate2 = Gate()
    assert not gate2.should_fire(GateSignals(button_pressed=False, cache_available=True))
    assert not gate2.should_fire(GateSignals(button_pressed=True, cache_available=False))
