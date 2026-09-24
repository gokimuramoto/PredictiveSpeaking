"""ASR partialの安定化。

直近N個のpartialの最長共通prefixを安定部分とし、巻き戻り(ASRの過去書き換え)を検出する。
予測は「stable + 不安定末尾」全体から行う設計のため、ここでの主な役割は
(1) KVチェックポイント境界の提供 (2) 巻き戻り検出 (3) 予測発火のペーシング。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def _lcp(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class PrefixUpdate:
    stable: str          # KVチェックポイント境界に相当する安定prefix
    tail: str            # 最新partialのうち安定部分より後ろ(毎サイクル貼り直す部分)
    latest: str          # 最新partial全体 (= stable + tail)
    rewound: bool        # 安定部分が縮んだ(ASRが過去を書き換えた)
    should_predict: bool # このタイミングで予測を発火すべきか


class StablePrefixTracker:
    def __init__(
        self,
        history_size: int = 3,
        min_growth_chars: int = 2,
        max_interval_ms: float = 400.0,
    ):
        self._recent: deque[str] = deque(maxlen=history_size)
        self._stable = ""
        self._last_fire_text = ""
        self._last_fire_t: float | None = None
        self.min_growth_chars = min_growth_chars
        self.max_interval_ms = max_interval_ms

    @property
    def stable(self) -> str:
        return self._stable

    def reset(self) -> None:
        self._recent.clear()
        self._stable = ""
        self._last_fire_text = ""
        self._last_fire_t = None

    def update(self, partial: str, t_ms: float) -> PrefixUpdate:
        self._recent.append(partial)

        candidate = self._recent[0]
        for p in list(self._recent)[1:]:
            candidate = _lcp(candidate, p)

        rewound = len(candidate) < len(self._stable)
        self._stable = candidate
        tail = partial[len(self._stable):]

        should = self._should_fire(partial, t_ms, rewound)
        if should:
            self._last_fire_text = partial
            self._last_fire_t = t_ms

        return PrefixUpdate(
            stable=self._stable,
            tail=tail,
            latest=partial,
            rewound=rewound,
            should_predict=should,
        )

    def _should_fire(self, partial: str, t_ms: float, rewound: bool) -> bool:
        if not partial:
            return False
        if rewound:
            return True
        if self._last_fire_t is None:
            return True
        growth = len(partial) - len(self._last_fire_text)
        if partial != self._last_fire_text and not partial.startswith(self._last_fire_text):
            return True  # 内容が書き換わった
        if growth >= self.min_growth_chars:
            return True
        if growth >= 1 and (t_ms - self._last_fire_t) >= self.max_interval_ms:
            return True
        return False
