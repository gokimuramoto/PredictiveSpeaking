"""Chatterbox-vLLM TTS server (英語TTS高速版、A6000で実行)。

randombk/chatterbox-vllm によるvLLM移植版。t3(トークン生成)が~4倍速になる
(s3gen波形生成はtorchのままなので短文単発の実効は~2倍想定)。
エンドポイントはIrodori互換(chatterbox_server.pyと同一)、ポート18084。
ボイスは既存 chatterbox_server/voices_en を共有する。

起動: PORT=18084 .venv/bin/python chatterbox_vllm_server.py
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

# 重要: このimportはトップレベルに置くこと。vLLMのspawn子プロセスは__main__を
# 再importするため、ここでchatterbox_vllmのimport副作用(EnTokenizerの
# TokenizerRegistry登録)が子プロセスでも実行される。関数内の遅延importにすると
# 子プロセスに登録が無く EngineCore が起動に失敗する
from chatterbox_vllm.tts import ChatterboxTTS  # noqa: E402

PORT = int(os.environ.get("PORT", "18084"))
# 共有GPUのため控えめに確保(0.15×49GB≈7.3GB)。torch版(18083)停止後に起動する前提
GPU_FRACTION = float(os.environ.get("GPU_FRACTION", "0.15"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "../chatterbox_server/voices_en")).resolve()
VOICES_DIR.mkdir(parents=True, exist_ok=True)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[cbx-vllm] loading model (gpu_frac={GPU_FRACTION}) ...", flush=True)
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization=GPU_FRACTION,
        max_model_len=1000,
        enforce_eager=True,
    )
    try:
        t0 = time.time()
        _ = model.generate(["Hello, this is a warmup."])
        print(f"[cbx-vllm] warmup done ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[cbx-vllm] warmup failed: {e}", flush=True)
    print("[cbx-vllm] ready", flush=True)
    yield


app = FastAPI(title="Chatterbox-vLLM EN TTS", lifespan=lifespan)


def _voice_path(voice_id: str) -> Path | None:
    for ext in (".wav", ".mp3", ".flac"):
        p = VOICES_DIR / f"{voice_id}{ext}"
        if p.exists():
            return p
    return None


def _sample_rate() -> int:
    return int(getattr(model, "sr", 24000))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "chatterbox-vllm-en",
            "voices": {"files": len(list(VOICES_DIR.glob('*')))}}


@app.get("/v1/audio/voices")
async def list_voices() -> dict:
    ids = sorted({p.stem for p in VOICES_DIR.glob("*") if p.is_file()})
    return {"object": "list",
            "data": [{"id": "default", "object": "voice"}] + [{"id": i, "object": "voice"} for i in ids]}


@app.post("/v1/audio/voices", status_code=201)
async def upload_voice(file: UploadFile = File(...), voice_id: str = Form(None)) -> dict:
    vid = (voice_id or Path(file.filename or "voice").stem).strip()[:32]
    raw = await file.read()
    (VOICES_DIR / f"{vid}.wav").write_bytes(raw)
    return {"id": vid, "object": "voice"}


@app.post("/v1/audio/speech")
async def speech(body: dict) -> Response:
    text = str(body.get("input", "")).strip()
    voice = str(body.get("voice", "default"))
    if not text:
        raise HTTPException(status_code=400, detail="input required")
    t0 = time.time()
    kwargs = {}
    if voice != "default":
        p = _voice_path(voice)
        if p is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
        kwargs["audio_prompt_path"] = str(p)
    try:
        audios = model.generate([text], **kwargs)
    except Exception as e:
        print(f"[cbx-vllm] generate error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    wav = audios[0]
    audio = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "squeeze") else wav
    rate = _sample_rate()
    gen_ms = (time.time() - t0) * 1000
    dur_ms = len(audio) / rate * 1000
    print(f"[cbx-vllm] '{text[:40]}' voice={voice} gen={gen_ms:.0f}ms "
          f"audio={dur_ms:.0f}ms rtf={gen_ms/max(dur_ms,1):.2f}", flush=True)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
