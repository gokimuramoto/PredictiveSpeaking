"""ストリーミングASRの共通インターフェース。

バックエンド(azure / vosk / typed)は差し替え可能。
すべてのバックエンドは push_audio() でPCMを受け取り、events() でASREventを流す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class ASREvent:
    text: str          # 現在発話のpartial全文(finalの場合は確定文)
    is_final: bool
    t_ms: float        # 受信時刻(monotonic)


class StreamingASR(Protocol):
    sample_rate: int

    async def start(self) -> None: ...

    def push_audio(self, pcm_s16le: bytes) -> None:
        """16kHz mono s16le のフレームを供給する(非ブロッキング)。"""
        ...

    def events(self) -> AsyncIterator[ASREvent]: ...

    async def stop(self) -> None: ...
