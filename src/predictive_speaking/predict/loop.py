"""常時予測ループ。

latest-value方式: 入力(最新partial)は常に上書きされ、ループは「今の最新」に対して
予測→publishを繰り返す。予測中に届いた入力はコアレスされ、完了後すぐ次サイクルが走る。
(進行中リクエストのキャンセルは行わない — cache_promptにより次サイクルの
 プリフィルは差分のみで安価なため、完了させて古さをstaleフラグで示す方が
 スループットと単純さで勝る。キャンセル戦略は将来の実験項目。)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from ..clock import now_ms
from ..events import EventLogger
from .base import Predictor
from .types import CandidateSet


@dataclass
class LoopStats:
    cycles: int = 0
    errors: int = 0
    stale_publishes: int = 0
    wall_ms: list[float] = field(default_factory=list)
    freshness_ms: list[float] = field(default_factory=list)  # 入力到着→候補publish


class PredictionLoop:
    def __init__(
        self,
        predictor: Predictor,
        on_result: Callable[[CandidateSet], None] | None = None,
        logger: EventLogger | None = None,
        min_cycle_ms: float = 40.0,
        error_backoff_ms: float = 300.0,
    ):
        self._predictor = predictor
        self._on_result = on_result
        self._log = logger or EventLogger(None)
        self._min_cycle_ms = min_cycle_ms
        self._error_backoff_ms = error_backoff_ms

        self._history = ""
        self._latest = ""
        self._version = 0
        self._input_t = 0.0
        self._new_input = asyncio.Event()
        self._stopping = False
        self._seq = 0

        self.stats = LoopStats()
        self.latest_result: CandidateSet | None = None

    # ---------- input side ----------

    def submit(self, history: str, latest_partial: str) -> None:
        """ASR側から最新状態を上書き投入する(スレッド外から呼ぶ場合はcall_soon_threadsafe)。"""
        self._version += 1
        self._history = history
        self._latest = latest_partial
        self._input_t = now_ms()
        self._new_input.set()

    def stop(self) -> None:
        self._stopping = True
        self._new_input.set()

    # ---------- loop ----------

    async def run(self) -> None:
        while True:
            await self._new_input.wait()
            self._new_input.clear()
            if self._stopping:
                return

            version = self._version
            history, latest, input_t = self._history, self._latest, self._input_t
            if not latest:
                continue

            self._seq += 1
            t0 = now_ms()
            self._log.log("predict_start", request_id=self._seq, input=latest)
            try:
                result = await self._predictor.predict(history, latest, self._seq)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats.errors += 1
                self._log.log("predict_error", request_id=self._seq, error=str(e))
                await asyncio.sleep(self._error_backoff_ms / 1000)
                continue

            stale = version != self._version
            result.extra["stale"] = stale
            result.extra["input_version"] = version
            freshness = now_ms() - input_t
            result.extra["freshness_ms"] = round(freshness, 1)

            self.latest_result = result
            self.stats.cycles += 1
            if stale:
                self.stats.stale_publishes += 1
            self.stats.wall_ms.append(result.wall_ms)
            self.stats.freshness_ms.append(freshness)

            best = result.best
            self._log.log(
                "candidates",
                request_id=result.request_id,
                input=result.input_latest,
                best=best.text if best else None,
                n=len(result.candidates),
                agree_len=result.agree_len,
                confidence=result.confidence,
                wall_ms=round(result.wall_ms, 1),
                freshness_ms=result.extra["freshness_ms"],
                stale=stale,
            )
            if self._on_result:
                self._on_result(result)

            elapsed = now_ms() - t0
            if elapsed < self._min_cycle_ms:
                await asyncio.sleep((self._min_cycle_ms - elapsed) / 1000)
