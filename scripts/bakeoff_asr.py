"""ASRベイクオフ: 同一wavを複数バックエンドに実時間ストリーミングして横並び比較。

  uv run python scripts/bakeoff_asr.py my_voice.wav                       # 既定: en-US, azure,vosk
  uv run python scripts/bakeoff_asr.py my_voice.wav ja-JP azure
"""

from __future__ import annotations

import asyncio
import audioop
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config  # noqa: E402

from predictive_speaking.clock import now_ms  # noqa: E402
from predictive_speaking.score import percentile  # noqa: E402


def load_wav_16k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    if ch == 2:
        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    return pcm


async def run_backend(name: str, asr, pcm: bytes) -> dict:
    await asr.start()
    partials: list[tuple[float, str]] = []
    finals: list[str] = []

    async def consume():
        async for ev in asr.events():
            if ev.is_final:
                finals.append(ev.text)
            elif ev.text:
                partials.append((now_ms(), ev.text))

    task = asyncio.create_task(consume())
    frame = 16000 * 2 * 20 // 1000
    t0 = now_ms()
    for i in range(0, len(pcm), frame):
        asr.push_audio(pcm[i:i + frame])
        await asyncio.sleep(0.02)
    await asyncio.sleep(2.5)
    await asr.stop()
    task.cancel()

    iv = [partials[i][0] - partials[i - 1][0] for i in range(1, len(partials))
          if partials[i][0] - partials[i - 1][0] < 2000]
    first_partial_ms = partials[0][0] - t0 if partials else None
    return {
        "name": name,
        "partials": len(partials),
        "first_partial_ms": round(first_partial_ms) if first_partial_ms else None,
        "interval_p50": round(percentile(iv, 50)) if iv else None,
        "interval_p90": round(percentile(iv, 90)) if iv else None,
        "text": "。".join(finals) if finals else (partials[-1][1] if partials else ""),
    }


async def main() -> None:
    config.load_env()
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/me_test.wav"
    pcm = load_wav_16k(path)
    print(f"audio: {len(pcm)/32000:.1f}s\n")

    lang = sys.argv[2] if len(sys.argv) > 2 else "en-US"
    names = (sys.argv[3] if len(sys.argv) > 3 else "azure,vosk").split(",")

    def build(name: str):
        if name == "azure":
            from predictive_speaking.asr.azure_speech import AzureSpeechASR

            return AzureSpeechASR(language=lang)
        if name == "vosk":
            from predictive_speaking.asr.vosk_asr import VoskASR

            return VoskASR(lang="en" if lang.startswith("en") else "ja")
        raise ValueError(f"unknown backend: {name}")

    for name, asr in [(n, build(n)) for n in names]:
        r = await run_backend(name, asr, pcm)
        print(f"== {r['name']} ==")
        print(f"  初回partial: {r['first_partial_ms']}ms  間隔p50: {r['interval_p50']}ms  p90: {r['interval_p90']}ms  (n={r['partials']})")
        print(f"  認識: {r['text'][:120]}")
        err = getattr(asr, "last_error", None)
        if err:
            print(f"  error: {err}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
