"""Chatterbox TTS server (英語版TTS、A6000で実行)。

エンドポイントはIrodori-TTS-Server互換(OpenAI /v1/audio/speech + voices管理)なので、
クライアントは既存の IrodoriTTSClient をURL差し替えだけで使える。

起動: PORT=18083 .venv/bin/python chatterbox_server.py
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

PORT = int(os.environ.get("PORT", "18083"))
VOICES_DIR = Path("./voices_en")
VOICES_DIR.mkdir(exist_ok=True)

model = None
_default_conds = None          # 内蔵声の条件(クローン声で上書きされるため退避)
_conds_cache: dict[str, object] = {}  # voice_id -> Conditionals(参照音声の条件を1回だけ計算)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, _default_conds
    from chatterbox.tts import ChatterboxTTS

    print("[chatterbox] loading model ...", flush=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    _default_conds = model.conds
    # ウォームアップ(初回コンパイル・CUDA初期化を吸収)
    try:
        t0 = time.time()
        _ = model.generate("Hello, this is a warmup.")
        print(f"[chatterbox] warmup done ({time.time()-t0:.1f}s, sr={model.sr})", flush=True)
    except Exception as e:
        print(f"[chatterbox] warmup failed: {e}", flush=True)
    print("[chatterbox] ready", flush=True)
    yield


def _voice_conds(voice: str, path: Path):
    """参照音声の条件付け(話者埋め込み・プロンプトトークン)を声ごとにキャッシュ。

    generate(audio_prompt_path=...)は毎回prepare_conditionalsをやり直すため、
    クローン声の合成に参照音声の前処理コストが毎回乗っていた。
    """
    conds = _conds_cache.get(voice)
    if conds is None:
        model.prepare_conditionals(str(path))
        conds = model.conds
        _conds_cache[voice] = conds
    return conds


app = FastAPI(title="Chatterbox EN TTS", lifespan=lifespan)


def _voice_path(voice_id: str) -> Path | None:
    for ext in (".wav", ".mp3", ".flac"):
        p = VOICES_DIR / f"{voice_id}{ext}"
        if p.exists():
            return p
    return None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "chatterbox-en",
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
    _conds_cache.pop(vid, None)  # 同名再登録時に古い条件を捨てる
    return {"id": vid, "object": "voice"}


@app.post("/v1/audio/speech")
async def speech(body: dict) -> Response:
    text = str(body.get("input", "")).strip()
    voice = str(body.get("voice", "default"))
    if not text:
        raise HTTPException(status_code=400, detail="input required")
    t0 = time.time()
    try:
        if voice != "default":
            p = _voice_path(voice)
            if p is None:
                raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
            model.conds = _voice_conds(voice, p)
        else:
            model.conds = _default_conds
        wav = model.generate(text)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[chatterbox] generate error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    audio = wav.squeeze().detach().cpu().numpy()
    gen_ms = (time.time() - t0) * 1000
    dur_ms = len(audio) / model.sr * 1000
    print(f"[chatterbox] '{text[:40]}' voice={voice} gen={gen_ms:.0f}ms "
          f"audio={dur_ms:.0f}ms rtf={gen_ms/max(dur_ms,1):.2f}", flush=True)
    buf = io.BytesIO()
    sf.write(buf, audio, model.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
