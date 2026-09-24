"""Piper (VITS, ONNX) のインプロセスTTS — 本人声fine-tuneモデルをこのPCのCPUで動かす。

KokoroTTSClient と同一インターフェース。モデルは models/piper/<voice>.onnx + <voice>.onnx.json
(GPUサーバでfine-tuneして書き出したもの: server/piper_train/)。
声の切替 = モデルファイルの切替(list_voices はディレクトリ内の .onnx を列挙)。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..clock import now_ms
from .irodori_client import _trim_silence

DEFAULT_MODEL_DIR = Path("models/piper")
STOCK_VOICES = {"en_US-lessac-medium"}  # 市販声(本人声モデルが無いときの既定)


class PiperTTSClient:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, voice: str = "",
                 length_scale: float | None = None):
        self.model_dir = Path(model_dir)
        self.voice = voice or self._first_voice()
        self.length_scale = length_scale  # 話速(1.0=学習時、小さいほど速い)
        self.sample_rate = 22050
        self._voices: dict[str, object] = {}
        self._lock = asyncio.Lock()

    # ---- モデル管理 ----
    def _first_voice(self) -> str:
        """既定の声: 本人声モデルを市販声(setup_models.pyが入れるlessac)より優先する。"""
        cands = sorted(p.stem for p in self.model_dir.glob("*.onnx"))
        own = [c for c in cands if c not in STOCK_VOICES]
        return (own or cands or [""])[0]

    def _load(self, voice: str):
        v = self._voices.get(voice)
        if v is None:
            from piper import PiperVoice

            onnx = self.model_dir / f"{voice}.onnx"
            cfg = self.model_dir / f"{voice}.onnx.json"
            if not onnx.exists():
                raise FileNotFoundError(f"Piperモデルがありません: {onnx}")
            v = PiperVoice.load(str(onnx), config_path=str(cfg) if cfg.exists() else None,
                                use_cuda=False)
            self._voices[voice] = v
        return v

    def _synth_blocking(self, text: str) -> tuple[bytes, int]:
        v = self._load(self.voice)
        rate = int(getattr(getattr(v, "config", None), "sample_rate", 22050))
        # piper1-gpl: synthesize() -> AudioChunk列 / 旧piper: synthesize_stream_raw()
        if hasattr(v, "synthesize") and not hasattr(v, "synthesize_stream_raw"):
            kwargs = {}
            if self.length_scale is not None:
                from piper import SynthesisConfig

                kwargs["syn_config"] = SynthesisConfig(length_scale=self.length_scale)
            parts = []
            for chunk in v.synthesize(text, **kwargs):
                parts.append(chunk.audio_int16_bytes)
                rate = int(getattr(chunk, "sample_rate", rate))
            return b"".join(parts), rate
        kwargs = {"length_scale": self.length_scale} if self.length_scale is not None else {}
        return b"".join(v.synthesize_stream_raw(text, **kwargs)), rate

    # ---- 共通インターフェース ----
    async def health(self) -> dict:
        await asyncio.get_running_loop().run_in_executor(None, self._load, self.voice)
        return {"status": "healthy", "backend": "piper-local", "voice": self.voice,
                "model_dir": str(self.model_dir)}

    async def probe(self) -> int:
        await self.synthesize("Hello.")
        return self.sample_rate

    async def synthesize(self, text: str, language: str = "English") -> tuple[bytes, float]:
        t0 = now_ms()
        async with self._lock:
            pcm, rate = await asyncio.get_running_loop().run_in_executor(
                None, self._synth_blocking, text)
        self.sample_rate = rate
        return _trim_silence(pcm, rate), now_ms() - t0

    async def list_voices(self) -> list[str]:
        return sorted(p.stem for p in self.model_dir.glob("*.onnx"))

    async def register_voice(self, wav_bytes: bytes, voice_id: str,
                             language: str = "English") -> dict:
        raise RuntimeError("Piperは学習済みモデル方式のため即時登録は非対応です "
                           "(server/piper_train/ のパイプラインでfine-tuneしてください)")

    async def close(self) -> None:
        self._voices.clear()
