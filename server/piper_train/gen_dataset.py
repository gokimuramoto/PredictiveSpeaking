"""Chatterboxクローン声でPiper学習用データセットを合成する(サーバ側で実行)。

CMU ARCTICプロンプト(音素バランス英文1132本、パブリックドメイン)を chatterbox-vllm の
HTTP API(/v1/audio/speech)で指定ボイスにて合成し、22050Hz/16bit monoのwavと
metadata.csv(LJSpeech形式: id|text)を出力する。長さ異常(幻聴・欠落)は除外する。

  .venv/bin/python gen_dataset.py --voice neshime --out data/neshime
  (再実行時は既存wavをスキップ=再開可能)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch
import torchaudio

ARCTIC_RE = re.compile(r'\(\s*(\S+)\s+"(.*)"\s*\)')


def load_prompts(path: Path) -> list[tuple[str, str]]:
    items = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ARCTIC_RE.match(line.strip())
        if m:
            items.append((m.group(1), m.group(2).strip()))
    return items


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -40.0, pad_ms: int = 60) -> np.ndarray:
    frame = max(1, sr // 100)  # 10ms
    n = len(x) // frame
    if n < 3:
        return x
    rms = np.array([np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    idx = np.where(db > thresh_db)[0]
    if len(idx) == 0:
        return x
    pad = sr * pad_ms // 1000
    a = max(0, idx[0] * frame - pad)
    b = min(len(x), (idx[-1] + 1) * frame + pad)
    return x[a:b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="cmuarctic.data")
    ap.add_argument("--url", default="http://127.0.0.1:18084")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"[gen] voice={args.voice} prompts={len(prompts)} -> {out}", flush=True)

    rejected_path = out / "rejected.jsonl"
    meta: dict[str, str] = {}
    meta_path = out / "metadata.csv"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="|"):
                if len(row) >= 2:
                    meta[row[0].removesuffix(".wav")] = row[1]

    client = httpx.Client(timeout=120.0)
    t_start = time.time()
    n_ok = n_rej = n_skip = 0
    for i, (uid, text) in enumerate(prompts, 1):
        wav_path = wav_dir / f"{uid}.wav"
        if wav_path.exists() and uid in meta:
            n_skip += 1
            continue
        try:
            r = client.post(f"{args.url}/v1/audio/speech",
                            json={"input": text, "voice": args.voice, "response_format": "wav"})
            r.raise_for_status()
            x, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        except Exception as e:
            with rejected_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": uid, "reason": f"http:{e}"}) + "\n")
            n_rej += 1
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != args.sr:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, args.sr).numpy()
        x = trim_silence(x, args.sr)
        dur = len(x) / args.sr
        cps = len(text) / max(dur, 1e-6)  # 文字/秒: 幻聴(長すぎ)や欠落(短すぎ)の検出
        if dur < 0.6 or dur > 12.0 or cps < 6 or cps > 28:
            with rejected_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": uid, "reason": f"dur={dur:.2f} cps={cps:.1f}"}) + "\n")
            n_rej += 1
            continue
        peak = float(np.max(np.abs(x)) or 1.0)
        if peak > 0.99:
            x = x / peak * 0.95
        sf.write(str(wav_path), x, args.sr, subtype="PCM_16")
        meta[uid] = text
        n_ok += 1
        if i % 25 == 0:
            el = time.time() - t_start
            print(f"[gen] {i}/{len(prompts)} ok={n_ok} rej={n_rej} skip={n_skip} "
                  f"({el/60:.1f}min, {el/max(1,i-n_skip):.2f}s/utt)", flush=True)
            with meta_path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
                for k in sorted(meta):
                    w.writerow([f"{k}.wav", meta[k]])  # piper1-gpl形式: audio_file.wav|text

    with meta_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for k in sorted(meta):
            w.writerow([f"{k}.wav", meta[k]])
    total_dur = 0.0
    for k in meta:
        p = wav_dir / f"{k}.wav"
        if p.exists():
            total_dur += sf.info(str(p)).duration
    print(f"[gen] DONE voice={args.voice} ok={len(meta)} rejected={n_rej} "
          f"total_audio={total_dur/60:.1f}min elapsed={(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
