"""IrodoriTTSクライアントの疎通スモーク。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.tts.irodori_client import IrodoriTTSClient  # noqa: E402


async def main() -> None:
    voice = sys.argv[1] if len(sys.argv) > 1 else "me"
    client = IrodoriTTSClient(voice=voice)
    try:
        print("health:", await client.health())
        for text in ["予測して音声で補助します", "次の言葉を", "ここで重要なのは先読みです"]:
            pcm, ms = await client.synthesize(text)
            dur = len(pcm) / 2 / client.sample_rate * 1000
            print(f"synth '{text}': {ms:.0f}ms, audio {dur:.0f}ms, rate {client.sample_rate}, rtf {ms/dur:.2f}")
        Path("logs/smoke_irodori.pcm").write_bytes(pcm)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
