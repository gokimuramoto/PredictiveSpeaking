"""マイク録音ヘルパー: ASRベイクオフ用のwavを作る(音量RMSも表示)。

  uv run python scripts/record_wav.py logs/en_real.wav --seconds 15
  uv run python scripts/bakeoff_asr.py logs/en_real.wav en-US azure,vosk
"""

from __future__ import annotations

import argparse
import audioop
import wave


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--rate", type=int, default=16000)
    args = ap.parse_args()

    import sounddevice as sd

    print(f"🔴 {args.seconds}秒間録音します。普段の調子で英語を話してください...")
    audio = sd.rec(int(args.seconds * args.rate), samplerate=args.rate,
                   channels=1, dtype="int16")
    sd.wait()
    pcm = audio.tobytes()
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.rate)
        wf.writeframes(pcm)
    rms = audioop.rms(pcm, 2)
    print(f"✅ 保存: {args.out}  RMS={rms}", end="")
    if rms < 500:
        print("  ⚠ 音量がかなり低い(マイク設定/距離を確認。ASR誤認識の主因になりえます)")
    elif rms < 1200:
        print("  ⚠ やや低め")
    else:
        print("  (音量OK)")


if __name__ == "__main__":
    main()
