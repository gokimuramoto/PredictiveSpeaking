"""JSONLイベントログ。全イベントにセッション開始基準のmonotonic t_msを付与する。"""

from __future__ import annotations

import json
import os
from typing import Any, TextIO

from .clock import now_ms


class EventLogger:
    def __init__(self, path: str | None):
        self._t0 = now_ms()
        self._fh: TextIO | None = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    @property
    def t_ms(self) -> float:
        return now_ms() - self._t0

    def log(self, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        rec = {"t_ms": round(self.t_ms, 1), "event": event, **fields}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 強制終了(ターミナルclose等)でセッション丸ごと消えるのを防ぐ。
        # イベントは数件/秒なので毎行flushのコストは無視できる
        self._fh.flush()

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
