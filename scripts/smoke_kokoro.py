"""Kokoroローカル TTSのスモーク: 合成レイテンシ/RTF計測 + wav保存。

  uv run python scripts/smoke_kokoro.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.tts.kokoro_client import KokoroTTSClient  # noqa: E402

TEXTS = [
    "and then we can move on to the next item.",
    "I think the main issue is the response latency.",
    "so let's start with the weekly progress update.",
]


async def main() -> None:
    client = KokoroTTSClient()
    print("[health]", await client.health())
    print("[voices]", await client.list_voices())
    out_dir = Path("logs/kokoro_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(TEXTS):
        t0 = time.perf_counter()
        pcm, _ = await client.synthesize(text)
        ms = (time.perf_counter() - t0) * 1000
        dur = len(pcm) / 2 / client.sample_rate * 1000
        out = out_dir / f"kokoro_{i}.wav"
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(client.sample_rate)
            wf.writeframes(pcm)
        print(f"[synth] #{i} {ms:.0f}ms audio={dur:.0f}ms rtf={ms / max(dur, 1):.2f} "
              f"sr={client.sample_rate} -> {out}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
