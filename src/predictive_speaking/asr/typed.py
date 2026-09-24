"""Typed ASR: テキストを文字刻みで流す決定的な疑似ASR。

音声なしでパイプライン全体(prefix追跡→予測→表示)の動作とfreshnessを測る。
ASRの揺らぎ(数文字ずつ・不定間隔)を粗く模擬する。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..clock import now_ms
from .base import ASREvent


class TypedASR:
    def __init__(self, script: list[str], chars_per_tick: int = 3, tick_ms: float = 300.0,
                 sentence_gap_ms: float = 1200.0, sample_rate: int = 16000,
                 stall_ms: float = 0.0, stall_frac: float = 0.6):
        self.sample_rate = sample_rate
        self.script = script
        self.chars_per_tick = chars_per_tick
        self.tick_ms = tick_ms
        self.sentence_gap_ms = sentence_gap_ms
        self.stall_ms = stall_ms          # >0なら各文の途中で言い淀み(partial凍結)を模擬
        self.stall_frac = stall_frac
        self._events: asyncio.Queue[ASREvent | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        for sentence in self.script:
            pos = 0
            stall_at = int(len(sentence) * self.stall_frac) if self.stall_ms > 0 else -1
            stalled = False
            while pos < len(sentence):
                pos = min(len(sentence), pos + self.chars_per_tick)
                await self._events.put(ASREvent(text=sentence[:pos], is_final=False, t_ms=now_ms()))
                if not stalled and stall_at >= 0 and pos >= stall_at:
                    stalled = True
                    await asyncio.sleep(self.stall_ms / 1000)  # 言い淀み: partial凍結
                await asyncio.sleep(self.tick_ms / 1000)
            await self._events.put(ASREvent(text=sentence, is_final=True, t_ms=now_ms()))
            await asyncio.sleep(self.sentence_gap_ms / 1000)
        await self._events.put(None)  # 終了合図

    def push_audio(self, pcm_s16le: bytes) -> None:
        pass  # 音声は使わない

    async def events(self) -> AsyncIterator[ASREvent]:
        while True:
            ev = await self._events.get()
            if ev is None:
                return
            yield ev

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
