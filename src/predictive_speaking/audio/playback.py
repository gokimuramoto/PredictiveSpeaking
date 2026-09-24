"""即時再生・即時停止のプレイバック。

常駐のOutputStreamコールバックがバッファから吸い出す方式:
- play() はバッファ差し替えのみ(体感遅延はブロックサイズ数十ms以下)
- stop() はバッファクリアのみ(<100ms停止の要件)
"""

from __future__ import annotations

import threading


class PlaybackController:
    def __init__(self, sample_rate: int = 24000, device: int | None = None, frame_ms: int = 20):
        self.sample_rate = sample_rate
        self.device = device
        self.frame_bytes = sample_rate * 2 * frame_ms // 1000
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self.playing_text: str | None = None

    def start(self) -> None:
        import sounddevice as sd

        def callback(outdata, frames, time_info, status) -> None:
            need = len(outdata)
            with self._lock:
                take = bytes(self._buf[:need])
                del self._buf[:need]
                if take and not self._buf:
                    # 最後まで自然に鳴り切った(手動stopと区別し、履歴コミットに使う)
                    self._completed_label = self.playing_text
                    self.playing_text = None
            outdata[: len(take)] = take
            if len(take) < need:
                outdata[len(take):] = b"\x00" * (need - len(take))

        # デバイス制約(MMEの多ch/レート制限等)に合わせてフォールバック:
        # mono 24k → mono 48k → stereo 48k → stereo 44.1k。play()側で変換する。
        last_err: Exception | None = None
        for rate, ch in ((self.sample_rate, 1), (48000, 1), (48000, 2), (44100, 2)):
            try:
                self._stream = sd.RawOutputStream(
                    samplerate=rate, blocksize=rate * ch * 2 * 10 // 1000,
                    device=self.device, dtype="int16", channels=ch, callback=callback)
                self._stream.start()
                self.device_rate = rate
                self.device_channels = ch
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"出力ストリームを開けません: {last_err}")

    def play(self, pcm_s16le: bytes, label: str | None = None) -> None:
        import audioop

        if getattr(self, "device_rate", self.sample_rate) != self.sample_rate:
            pcm_s16le, _ = audioop.ratecv(
                pcm_s16le, 2, 1, self.sample_rate, self.device_rate, None)
        if getattr(self, "device_channels", 1) == 2:
            pcm_s16le = audioop.tostereo(pcm_s16le, 2, 1.0, 1.0)
        with self._lock:
            self._buf.clear()
            self._buf.extend(pcm_s16le)
            self.playing_text = label

    def stop(self) -> None:
        with self._lock:
            self._buf.clear()
            self.playing_text = None
            self._completed_label = None  # 途中停止はコミットしない

    def pop_completed(self) -> str | None:
        """自然に鳴り切った再生のラベルを1回だけ返す(履歴コミット用)。"""
        with self._lock:
            label = getattr(self, "_completed_label", None)
            self._completed_label = None
            return label

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def close(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
