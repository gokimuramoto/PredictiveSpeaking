"""予測精度・遅延の採点。

正解ラベルは自動: ある時点の予測に対し、その後実際に発話された続きと突き合わせる。
リプレイベンチでは全文が既知なので text[pos:] が正解。実運用でも後続ASRから同じ計算ができる。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .text.ja import normalize_for_match


def leading_match_len(pred: str, truth: str, cap: int = 8) -> int:
    """正規化(空白・句読点除去)後の先頭一致文字数。"""
    p = normalize_for_match(pred)
    t = normalize_for_match(truth)
    n = min(len(p), len(t), cap)
    i = 0
    while i < n and p[i] == t[i]:
        i += 1
    return i


@dataclass
class Row:
    fixture_id: str
    pos: int
    input_latest: str
    truth: str
    pred: str
    match_len: int
    speakable: bool
    silence: bool
    flags: list[str]
    confidence: float
    agree_len: int
    wall_ms: float
    prompt_ms: float | None
    predict_ms: float | None
    prompt_n: int | None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))
    return s[idx]


def summarize(rows: list[Row]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    spoken = [r for r in rows if r.speakable and r.pred]
    hit = lambda k: sum(1 for r in spoken if r.match_len >= k)
    walls = [r.wall_ms for r in rows]
    prompts = [r.prompt_ms for r in rows if r.prompt_ms is not None]
    predicts = [r.predict_ms for r in rows if r.predict_ms is not None]
    prompt_ns = [r.prompt_n for r in rows if r.prompt_n is not None]
    flags: dict[str, int] = {}
    for r in rows:
        for f in r.flags:
            flags[f] = flags.get(f, 0) + 1
    return {
        "n": n,
        "speak_rate": round(len(spoken) / n, 3),
        "silence_rate": round(sum(1 for r in rows if r.silence) / n, 3),
        "hit@1": round(hit(1) / max(1, len(spoken)), 3),
        "hit@2": round(hit(2) / max(1, len(spoken)), 3),
        "hit@3": round(hit(3) / max(1, len(spoken)), 3),
        "mean_match_len": round(sum(r.match_len for r in spoken) / max(1, len(spoken)), 2),
        "wall_ms": {"p50": round(percentile(walls, 50), 1), "p95": round(percentile(walls, 95), 1)},
        "prompt_ms": {"p50": round(percentile(prompts, 50), 1), "p95": round(percentile(prompts, 95), 1)} if prompts else None,
        "predict_ms": {"p50": round(percentile(predicts, 50), 1), "p95": round(percentile(predicts, 95), 1)} if predicts else None,
        "prompt_n_median": median(prompt_ns) if prompt_ns else None,
        "flags": flags,
    }
