"""Vosk ストリーミングASR(完全ローカル、ja/en)。

モデル既定: ja=models/vosk-model-ja-0.22 /
en=models/vosk-model-en-us-0.22-lgraph
認識はブロッキングのためexecutorスレッドで回し、結果をasyncio Queueへ渡す。
"""

from __future__ import annotations

import asyncio
import json
import queue as thread_queue
from pathlib import Path
from typing import AsyncIterator

from ..clock import now_ms
from .base import ASREvent

# repo/src/predictive_speaking/asr/vosk_asr.py → parents[3]=repo
_REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = str(_REPO / "models" / "vosk-model-ja-0.22")
DEFAULT_MODEL_PATH_EN = str(_REPO / "models" / "vosk-model-en-us-0.22-lgraph")


class VoskASR:
    def __init__(self, model_path: str | None = None, sample_rate: int = 16000,
                 lang: str = "ja"):
        self.sample_rate = sample_rate
        self.lang = lang
        self.model_path = model_path or (
            DEFAULT_MODEL_PATH_EN if lang == "en" else DEFAULT_MODEL_PATH)
        self._audio_q: thread_queue.Queue[bytes | None] = thread_queue.Queue(maxsize=200)
        self._events: asyncio.Queue[ASREvent] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._last_partial = ""

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()
        self._task = loop.run_in_executor(None, self._run_blocking, loop)  # type: ignore[assignment]
        # モデルロード完了まで待つ(en-lgraphで~5秒。ロード中にマイク音声が
        # キュー(4秒分)からあふれて冒頭が欠落するのを防ぐ)
        await self._ready.wait()

    def _run_blocking(self, loop: asyncio.AbstractEventLoop) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, self.sample_rate)
        loop.call_soon_threadsafe(self._ready.set)
        loop.call_soon_threadsafe(
            self._events.put_nowait, ASREvent(text="", is_final=False, t_ms=now_ms())
        )  # ready合図(空partial)
        # 日本語モデルはトークン間スペースを入れてくるため除去する。英語は保持
        strip = (lambda s: s.replace(" ", "")) if self.lang == "ja" else (lambda s: s)
        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                break
            if rec.AcceptWaveform(chunk):
                text = strip(json.loads(rec.Result()).get("text", ""))
                self._last_partial = ""
                if text:
                    loop.call_soon_threadsafe(
                        self._events.put_nowait, ASREvent(text=text, is_final=True, t_ms=now_ms())
                    )
            else:
                partial = strip(json.loads(rec.PartialResult()).get("partial", ""))
                if partial and partial != self._last_partial:
                    self._last_partial = partial
                    loop.call_soon_threadsafe(
                        self._events.put_nowait, ASREvent(text=partial, is_final=False, t_ms=now_ms())
                    )

    def push_audio(self, pcm_s16le: bytes) -> None:
        try:
            self._audio_q.put_nowait(pcm_s16le)
        except thread_queue.Full:
            pass  # 認識が追いつかない場合は古いフレームを落とすより新規を捨てない方が良いが、まずは単純に

    async def events(self) -> AsyncIterator[ASREvent]:
        while True:
            ev = await self._events.get()
            yield ev

    async def stop(self) -> None:
        self._audio_q.put(None)
