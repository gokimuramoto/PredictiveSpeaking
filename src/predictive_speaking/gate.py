"""介入ゲート。

予測エンジンが吐くシグナルを入力に「今、音を出すか」を決める純粋関数(policy)と、
その実行機構。プロトタイプのpolicyは手動ボタンのみ。将来は無音時間・VAP・
確信度などの条件をpolicy差し替えで追加する(予測側は変更しない)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .clock import now_ms


@dataclass
class GateSignals:
    button_pressed: bool = False
    cache_available: bool = False
    cache_age_ms: float = 0.0
    confidence: float = 0.0
    playing: bool = False
    cooldown_remaining_ms: float = 0.0
    # 将来: silence_ms, vap_continue_prob, prefix_incomplete, pplgap ...


Policy = Callable[[GateSignals], bool]


def button_policy(s: GateSignals) -> bool:
    """プロトタイプ: ボタンが押されていて、キャッシュがあり、再生中でなければ話す。"""
    return s.button_pressed and s.cache_available and not s.playing and s.cooldown_remaining_ms <= 0


@dataclass
class GateState:
    fired_count: int = 0
    last_fire_t_ms: float = field(default=0.0)


class Gate:
    def __init__(self, policy: Policy = button_policy, cooldown_ms: float = 600.0):
        self.policy = policy
        self.cooldown_ms = cooldown_ms
        self.state = GateState()

    def cooldown_remaining(self) -> float:
        if self.state.last_fire_t_ms == 0.0:
            return 0.0
        return max(0.0, self.cooldown_ms - (now_ms() - self.state.last_fire_t_ms))

    def should_fire(self, signals: GateSignals) -> bool:
        signals.cooldown_remaining_ms = self.cooldown_remaining()
        if self.policy(signals):
            self.state.fired_count += 1
            self.state.last_fire_t_ms = now_ms()
            return True
        return False
