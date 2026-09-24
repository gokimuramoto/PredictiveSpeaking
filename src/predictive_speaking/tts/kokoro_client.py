"""Kokoro-82M (ONNX) のインプロセスTTS — 英語オールローカル構成用。

サーバ不要でこのマシンのCPUで合成する(RTF~0.3, 24kHz)。IrodoriTTSClientと
同一インターフェース(health/probe/synthesize/list_voices/register_voice/close)。
声のクローンには非対応(固定ボイス) — クローンが必要な場合は遠隔Chatterboxを使う。

モデル配置: models/kokoro/kokoro-v1.0.int8.onnx + voices-v1.0.bin
(https://github.com/thewh1teagle/kokoro-onnx/releases の model-files-v1.0)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..clock import now_ms
from .irodori_client import _trim_silence

DEFAULT_MODEL_DIR = Path("models/kokoro")
# 品質の良い英語ボイスに絞って公開する(全ボイスは50以上あり選びにくい)
CURATED_VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]


class KokoroTTSClient:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.sample_rate = 24000  # probe()で実測値に更新される
        d = Path(model_dir)
        # fp32優先: DirectML(iGPU)互換が高く、AVX512機ではCPUでもint8より速い。
        # int8はfp32が無い場合のフォールバック(DML非対応・CPUでRTF~1.0)
        fp32 = d / "kokoro-v1.0.onnx"
        self._model_path = fp32 if fp32.exists() else d / "kokoro-v1.0.int8.onnx"
        self._voices_path = d / "voices-v1.0.bin"
        self._kokoro = None
        self._lock = asyncio.Lock()  # ONNXセッションへの合成要求を直列化

    def _ensure_loaded(self):
        if self._kokoro is None:
            if not self._model_path.exists() or not self._voices_path.exists():
                raise FileNotFoundError(
                    f"Kokoroモデルがありません: {self._model_path} / {self._voices_path}\n"
                    "README(英語オールローカル)のダウンロード手順を参照してください")
            import os

            # DirectMLはKokoroの一部opで実行時クラッシュするためCPUに固定する。
            # このCPU(AVX512)ではfp32+CPUがRTF~0.25と十分速い(int8は逆に~1.0)
            os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
        return self._kokoro

    async def health(self) -> dict:
        k = await asyncio.get_running_loop().run_in_executor(None, self._ensure_loaded)
        return {"status": "healthy", "backend": "kokoro-local", "voice": self.voice,
                "model": self._model_path.name, "voices": len(k.get_voices())}

    async def probe(self) -> int:
        await self.synthesize("Hello.")
        return self.sample_rate

    async def synthesize(self, text: str, language: str = "English") -> tuple[bytes, float]:
        t0 = now_ms()

        def _synth() -> tuple[bytes, int]:
            k = self._ensure_loaded()
            samples, rate = k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
            import numpy as np

            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return pcm, rate

        async with self._lock:
            pcm, rate = await asyncio.get_running_loop().run_in_executor(None, _synth)
        self.sample_rate = rate
        return _trim_silence(pcm, rate), now_ms() - t0

    async def list_voices(self) -> list[str]:
        k = await asyncio.get_running_loop().run_in_executor(None, self._ensure_loaded)
        available = set(k.get_voices())
        curated = [v for v in CURATED_VOICES if v in available]
        return curated or sorted(available)

    async def register_voice(self, wav_bytes: bytes, voice_id: str,
                             language: str = "English") -> dict:
        raise RuntimeError("Kokoroは声のクローン(登録)非対応です。クローンが必要な場合は "
                           "--tts-backend chatterbox(要サーバ接続)を使ってください")

    async def close(self) -> None:
        self._kokoro = None
