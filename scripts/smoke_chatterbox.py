"""Chatterbox EN TTSサーバのスモーク: health→声一覧→登録→合成レイテンシ計測。

  uv run python scripts/smoke_chatterbox.py                          # .envのPS_SERVER_HOSTS先頭:18084
  uv run python scripts/smoke_chatterbox.py --register my_voice.wav --voice-id me_en
  uv run python scripts/smoke_chatterbox.py --voices alice,bob       # 登録済みの声で合成のみ
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config  # noqa: E402
from predictive_speaking.tts.irodori_client import IrodoriTTSClient  # noqa: E402

config.load_env()

TEXTS = [
    "and then we can move on to the next item.",
    "I think the main issue is the response latency.",
    "so let's start with the weekly progress update.",
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"http://{config.server_host()}:18084",
                    help="既定: chatterbox-vllm(18084)。torch版は :18083")
    ap.add_argument("--register", default=None, help="参照wavを登録してから試す")
    ap.add_argument("--voice-id", default="me_en")
    ap.add_argument("--out", default="logs/chatterbox_smoke")
    ap.add_argument("--voices", default=None,
                    help="登録済み声IDをカンマ区切りで指定(登録なしで合成のみ。例: neshime,muramoto)")
    args = ap.parse_args()

    client = IrodoriTTSClient(base_url=args.url, voice="default")
    try:
        print("[health]", await client.health())
        if args.register:
            wav = Path(args.register).read_bytes()
            res = await client.register_voice(wav, voice_id=args.voice_id)
            print("[register]", res)
        voices = await client.list_voices()
        print("[voices]", voices)

        if args.voices:
            targets = [v.strip() for v in args.voices.split(",") if v.strip()]
        else:
            targets = ["default"] + ([args.voice_id] if args.register else [])
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for voice in targets:
            client.voice = voice
            for i, text in enumerate(TEXTS):
                t0 = time.perf_counter()
                pcm, _server_ms = await client.synthesize(text, language="English")
                ms = (time.perf_counter() - t0) * 1000
                rate = client.sample_rate
                dur_ms = len(pcm) / 2 / rate * 1000
                out = Path(args.out) / f"{voice}_{i}.wav"
                import wave

                with wave.open(str(out), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    wf.writeframes(pcm)
                print(f"[synth] voice={voice} #{i} {ms:.0f}ms audio={dur_ms:.0f}ms "
                      f"rtf={ms / max(dur_ms, 1):.2f} sr={rate} -> {out}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
