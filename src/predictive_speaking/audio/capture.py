"""音声入力ソース。

- MicSource: sounddeviceでマイクから16kHz mono s16leフレームを取得。
  コールバックはASRへのpushのみ(音声スレッドをブロックしない)。
- WavSource: wavファイルを実時間ペースで流す(再現テスト用)。
"""

from __future__ import annotations

import asyncio
import audioop
import wave


class MicSource:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, device: int | None = None):
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.device = device
        self._stream = None
        self.level = 0.0  # 直近フレームのRMS(表示用)

    def start(self, on_frame) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:
            pcm = bytes(indata)
            self.level = audioop.rms(pcm, 2) / 32768.0
            on_frame(pcm)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.frame_samples,
            device=self.device, dtype="int16", channels=1, callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()


class WavSource:
    def __init__(self, path: str, sample_rate: int = 16000, frame_ms: int = 20, realtime: bool = True):
        self.path = path
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.realtime = realtime
        self.level = 0.0

    async def run(self, on_frame) -> None:
        with wave.open(self.path, "rb") as wf:
            src_rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            state = None
            frame_src = src_rate * self.frame_ms // 1000
            while True:
                data = wf.readframes(frame_src)
                if not data:
                    break
                if channels == 2:
                    data = audioop.tomono(data, width, 0.5, 0.5)
                if width != 2:
                    data = audioop.lin2lin(data, width, 2)
                if src_rate != self.sample_rate:
                    data, state = audioop.ratecv(data, 2, 1, src_rate, self.sample_rate, state)
                self.level = audioop.rms(data, 2) / 32768.0
                on_frame(data)
                if self.realtime:
                    await asyncio.sleep(self.frame_ms / 1000)
