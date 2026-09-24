"""リプレイベンチ。

step mode: fixtureの全文を先頭から少しずつ見せ、各位置で予測→正解(残り全文)と採点。
精度はハード非依存、遅延はバックエンド依存。cache_promptの効果はprompt_n(プリフィル
トークン数)が2サイクル目以降で小さくなることで観測できる。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from ..predict.base import Predictor
from ..score import Row, leading_match_len, summarize


def load_fixtures(path: str, max_items: int | None = None) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_items and len(items) >= max_items:
                break
    return items


def step_positions(text: str, min_prefix: int = 5, stride: int = 3, max_steps: int = 8) -> list[int]:
    """予測を行うprefix位置。文頭すぎる位置は情報がないので除外。

    スペース区切り言語(英語)は単語境界に切る — 英語ASRのpartialは語単位で進むため。
    """
    if " " in text:
        word_ends = [i for i, ch in enumerate(text) if ch == " "]
        positions = [i for i in word_ends[1:] if i >= min_prefix]
        if len(positions) > max_steps:
            step = len(positions) / max_steps
            positions = [positions[int(k * step)] for k in range(max_steps)]
        return positions[:max_steps]
    positions = list(range(min_prefix, max(min_prefix + 1, len(text) - 2), stride))
    return positions[:max_steps]


async def run_fixture(
    predictor: Predictor, item: dict[str, Any], *,
    min_prefix: int, stride: int, max_steps: int, request_base: int,
) -> list[Row]:
    text: str = item["text"]
    history: str = item.get("history", "")
    rows: list[Row] = []
    for j, pos in enumerate(step_positions(text, min_prefix, stride, max_steps)):
        partial = text[:pos]
        truth = text[pos:]
        cs = await predictor.predict(history, partial, request_base + j)
        best = cs.best
        pred = best.text if best else ""
        rows.append(Row(
            fixture_id=str(item.get("id", "?")),
            pos=pos,
            input_latest=partial,
            truth=truth,
            pred=pred,
            match_len=leading_match_len(pred, truth) if pred else 0,
            speakable=bool(best and best.speakable),
            silence=any(c.silence for c in cs.candidates),
            flags=sorted({f for c in cs.candidates for f in c.flags}),
            confidence=cs.confidence,
            agree_len=cs.agree_len,
            wall_ms=cs.wall_ms,
            prompt_ms=best.prompt_ms if best else None,
            predict_ms=best.predict_ms if best else None,
            prompt_n=best.prompt_n if best else None,
        ))
    return rows


async def run_bench(
    predictor: Predictor,
    fixtures_path: str,
    *,
    min_prefix: int = 5,
    stride: int = 4,
    max_steps: int = 6,
    max_items: int | None = None,
    out_path: str | None = None,
) -> dict:
    items = load_fixtures(fixtures_path, max_items)
    all_rows: list[Row] = []
    for i, item in enumerate(items):
        rows = await run_fixture(
            predictor, item,
            min_prefix=min_prefix, stride=stride, max_steps=max_steps,
            request_base=i * 1000,
        )
        all_rows.extend(rows)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in all_rows:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    report = summarize(all_rows)
    report["fixtures"] = len(items)
    return report


def format_report(title: str, report: dict) -> str:
    lines = [f"== {title} =="]
    for k, v in report.items():
        if k == "flags":
            lines.append(f"  flags: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
