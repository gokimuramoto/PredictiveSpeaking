"""ASRバックエンドの疎通スモーク(無音2秒を流して起動・接続を確認する)。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402


async def smoke(name: str) -> None:
    silence = b"\x00" * (16000 * 2 // 50)  # 20ms
    if name == "vosk":
        from predictive_speaking.asr.vosk_asr import VoskASR

        asr = VoskASR()
        print(f"[vosk] model: {asr.model_path}")
    else:
        from predictive_speaking.asr.azure_speech import AzureSpeechASR

        asr = AzureSpeechASR()

    await asr.start()
    print(f"[{name}] started")
    got: list[str] = []

    async def consume():
        async for ev in asr.events():
            got.append(ev.text)

    task = asyncio.create_task(consume())
    for _ in range(100):  # 2秒
        asr.push_audio(silence)
        await asyncio.sleep(0.02)
    await asr.stop()
    task.cancel()
    print(f"[{name}] OK (events on silence: {len([g for g in got if g])})")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(smoke(sys.argv[1] if len(sys.argv) > 1 else "vosk"))
