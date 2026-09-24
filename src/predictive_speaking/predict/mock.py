"""テスト・オフライン開発用のMock Predictor。"""

from __future__ import annotations

import asyncio

from ..clock import now_ms
from ..validate import validate
from .types import Candidate, CandidateSet


class MockPredictor:
    """既知の全文からの「正解の続き」を返す(oracle)か、固定テキストを返す。

    oracle_text を与えると、latest_partial が oracle_text の先頭部分に一致する場合に
    その続き(最大max_chars)を返す — ループ/ベンチ配管の検証用。
    """

    def __init__(self, delay_ms: float = 50.0, oracle_text: str | None = None,
                 fixed: str = "続きです", max_chars: int = 15):
        self.delay_ms = delay_ms
        self.oracle_text = oracle_text
        self.fixed = fixed
        self.max_chars = max_chars
        self.calls = 0

    async def predict(self, history: str, latest_partial: str, request_id: int) -> CandidateSet:
        self.calls += 1
        t0 = now_ms()
        await asyncio.sleep(self.delay_ms / 1000)
        if self.oracle_text and self.oracle_text.startswith(latest_partial):
            raw = self.oracle_text[len(latest_partial):][: self.max_chars]
        else:
            raw = self.fixed
        verdict = validate(raw, latest_partial, self.max_chars)
        cand = Candidate(text=verdict.text, raw_text=raw, source="greedy",
                         speakable=verdict.speakable, flags=verdict.flags,
                         mean_logprob=-0.1, gen_ms=now_ms() - t0)
        return CandidateSet(
            request_id=request_id, input_latest=latest_partial,
            created_t_ms=t0, candidates=[cand], wall_ms=now_ms() - t0,
            agree_len=len(verdict.text), confidence=0.9,
        )

    async def close(self) -> None:
        return None
