"""Piperローカル TTS(本人声fine-tuneモデル)のスモーク: 合成レイテンシ/RTF計測 + wav保存。

  uv run python scripts/smoke_piper.py                       # models/piper の全モデル
  uv run python scripts/smoke_piper.py --voice en_US-neshime-medium
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.tts.piper_client import PiperTTSClient  # noqa: E402

TEXTS = [
    "and then we can move on to the next item.",
    "I think the main issue is the response latency.",
    "so let's start with the weekly progress update.",
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/piper")
    ap.add_argument("--voice", default=None, help="省略時はディレクトリ内の全モデル")
    ap.add_argument("--out", default="logs/piper_smoke")
    args = ap.parse_args()

    client = PiperTTSClient(model_dir=args.model_dir)
    voices = [args.voice] if args.voice else await client.list_voices()
    if not voices:
        print(f"モデルがありません: {args.model_dir}/*.onnx")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        client.voice = voice
        print("[health]", await client.health())
        for i, text in enumerate(TEXTS):
            t0 = time.perf_counter()
            pcm, _ = await client.synthesize(text)
            ms = (time.perf_counter() - t0) * 1000
            dur = len(pcm) / 2 / client.sample_rate * 1000
            out = out_dir / f"{voice}_{i}.wav"
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(client.sample_rate)
                wf.writeframes(pcm)
            print(f"[synth] {voice} #{i} {ms:.0f}ms audio={dur:.0f}ms "
                  f"rtf={ms / max(dur, 1):.2f} sr={client.sample_rate} -> {out}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
