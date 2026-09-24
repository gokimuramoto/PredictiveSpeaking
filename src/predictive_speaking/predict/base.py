"""Predictorインターフェース。バックエンド(llama-server / クラウド / mock)を差し替え可能にする。"""

from __future__ import annotations

from typing import Protocol

from .types import CandidateSet


class Predictor(Protocol):
    async def predict(self, history: str, latest_partial: str, request_id: int) -> CandidateSet:
        """historyと最新partialから継続候補セットを生成する。キャンセルはasyncioのcancelで行う。"""
        ...

    async def close(self) -> None: ...
