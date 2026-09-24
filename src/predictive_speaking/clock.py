"""monotonic時刻。すべての計測はwall clockではなくこれを使う。"""

import time


def now_ms() -> float:
    return time.perf_counter_ns() / 1e6
