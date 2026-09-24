"""Irodori-TTS-Server (OpenAI互換 /v1/audio/speech) クライアント。

サーバ: Irodori-TTS-Server (GPUサーバ, port 8088)
- 参照音声クローン: 登録済みボイスID(例: "me")を指定
- ストリーミング非対応(全量レスポンス) — 先行合成キャッシュ設計とは整合
- response_format=wav で受け、ヘッダからサンプルレートを自動検出する
"""

from __future__ import annotations

import io
import wave

import httpx

from ..clock import now_ms


def _trim_silence(pcm_s16le: bytes, rate: int,
                  threshold: int = 250, keep_head_ms: int = 40, keep_tail_ms: int = 120) -> bytes:
    """先頭・末尾の無音をトリムする(再生立ち上がりと間延びの改善)。"""
    import audioop

    frame = max(2, rate * 2 * 20 // 1000)  # 20ms
    n = len(pcm_s16le) // frame
    if n < 4:
        return pcm_s16le
    first, last = 0, n - 1
    while first < n and audioop.rms(pcm_s16le[first * frame:(first + 1) * frame], 2) < threshold:
        first += 1
    while last > first and audioop.rms(pcm_s16le[last * frame:(last + 1) * frame], 2) < threshold:
        last -= 1
    if first >= last:
        return pcm_s16le
    head = max(0, first * frame - rate * 2 * keep_head_ms // 1000)
    tail = min(len(pcm_s16le), (last + 1) * frame + rate * 2 * keep_tail_ms // 1000)
    return pcm_s16le[head:tail]


class IrodoriTTSClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8088",
        voice: str = "me",
        model: str = "irodori-tts",
        speed: float = 1.0,
        timeout_s: float = 60.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.model = model
        self.speed = speed
        self.sample_rate = 44100  # probe()で実測値に更新される
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Windows httpx→uvicornでkeep-alive再利用POSTやstream()が詰まる問題を回避(新規接続を使う)
        self._client = httpx.AsyncClient(
            timeout=timeout_s, headers=headers,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    async def health(self) -> dict:
        r = await self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {"status": "healthy", "backend": "irodori", "voice": self.voice, **data}

    async def probe(self) -> int:
        """短文を1回合成してサンプルレートを確定する(起動時に呼ぶ)。"""
        await self.synthesize("こんにちは")
        return self.sample_rate

    async def synthesize(self, text: str, language: str = "Japanese") -> tuple[bytes, float]:
        t0 = now_ms()
        r = await self._client.post(
            f"{self.base_url}/v1/audio/speech",
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
                "speed": self.speed,
            },
        )
        r.raise_for_status()
        with wave.open(io.BytesIO(r.content), "rb") as wf:
            self.sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())
        if channels == 2:
            import audioop

            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
        return _trim_silence(pcm, self.sample_rate), now_ms() - t0

    async def list_voices(self) -> list[str]:
        r = await self._client.get(f"{self.base_url}/v1/audio/voices")
        r.raise_for_status()
        return [v["id"] for v in r.json().get("data", [])]

    async def register_voice(self, wav_bytes: bytes, voice_id: str,
                             language: str = "Japanese") -> dict:
        """新しい参照音声をアップロードする(voice_id=ボイス名)。"""
        r = await self._client.post(
            f"{self.base_url}/v1/audio/voices",
            files={"file": (f"{voice_id}.wav", wav_bytes, "audio/wav")},
            data={"voice_id": voice_id},
        )
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()
