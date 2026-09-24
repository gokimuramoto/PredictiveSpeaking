import asyncio

import pytest

from predictive_speaking.predict.loop import PredictionLoop
from predictive_speaking.predict.mock import MockPredictor

ORACLE = "このシステムではユーザーが言葉に詰まったときに次の言葉を予測して音声で補助します"


@pytest.mark.asyncio
async def test_loop_publishes_latest():
    pred = MockPredictor(delay_ms=30, oracle_text=ORACLE)
    results = []
    loop = PredictionLoop(pred, on_result=results.append, min_cycle_ms=0)
    task = asyncio.create_task(loop.run())

    loop.submit("", ORACLE[:8])
    await asyncio.sleep(0.1)
    loop.submit("", ORACLE[:12])
    await asyncio.sleep(0.1)

    loop.stop()
    await asyncio.wait_for(task, timeout=1)

    assert len(results) >= 2
    last = results[-1]
    assert last.input_latest == ORACLE[:12]
    assert last.best is not None
    # oracleの続きと一致するはず
    assert ORACLE[12:].startswith(last.best.text[:3])


@pytest.mark.asyncio
async def test_loop_coalesces_rapid_inputs():
    """予測より速い入力連打でも、最後の入力に対する結果が最終的に出る。"""
    pred = MockPredictor(delay_ms=50, oracle_text=ORACLE)
    results = []
    loop = PredictionLoop(pred, on_result=results.append, min_cycle_ms=0)
    task = asyncio.create_task(loop.run())

    for i in range(6, 20, 2):
        loop.submit("", ORACLE[:i])
        await asyncio.sleep(0.01)  # 10ms間隔 << 50ms予測時間

    await asyncio.sleep(0.3)
    loop.stop()
    await asyncio.wait_for(task, timeout=1)

    # 全部は処理されない(コアレスされる)が、最後の入力は必ず処理される
    assert pred.calls < 8
    assert results[-1].input_latest == ORACLE[:18]
    assert loop.stats.cycles == len(results)


@pytest.mark.asyncio
async def test_loop_survives_backend_error():
    class FailingPredictor(MockPredictor):
        async def predict(self, history, latest_partial, request_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return await super().predict(history, latest_partial, request_id)

    pred = FailingPredictor(delay_ms=5, oracle_text=ORACLE)
    results = []
    loop = PredictionLoop(pred, on_result=results.append, min_cycle_ms=0, error_backoff_ms=10)
    task = asyncio.create_task(loop.run())

    loop.submit("", ORACLE[:8])
    await asyncio.sleep(0.05)
    loop.submit("", ORACLE[:10])
    await asyncio.sleep(0.1)

    loop.stop()
    await asyncio.wait_for(task, timeout=1)
    assert loop.stats.errors == 1
    assert results  # エラー後も回復して結果を出す
