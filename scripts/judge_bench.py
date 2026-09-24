"""ベンチ結果行をローカルLLMジャッジで評価する。

評価軸:
- coherence (0-2): 候補が「ここまでの文脈の流れに即しているか」。0=明確な逸脱。
- intent (0-2): 候補と「実際に話者が続けた内容」の意図一致。

hit@kが測れない「字面は違うが流れに即している/意図は同じ」を数値化する。
ジャッジは生成モデルとは別のinstructモデル(例: Swallow 8B SFT)を使うこと。

例:
  uv run python scripts/judge_bench.py --rows logs/bench_raw_swallow8b_q8_n3.jsonl \
    --judge-url http://<server>:18081 --out logs/judged_swallow8b.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config  # noqa: E402

config.load_env()

JUDGE_SYSTEM = """あなたは日本語の発話継続候補の評価者です。話者が発話の途中で言葉に詰まった際、システムが代わりに短い続きを発話します。その候補を評価してください。

[coherence] 候補が「ここまでの文脈の流れ」に即しているか:
2: 話題・語り口・既出内容と整合し、聞き手が違和感を持たない
1: 大きな破綻はないが、やや唐突・不自然・ずれがある
0: 話題の逸脱、既出内容との矛盾、意味不明、会話相手への返答になっている等の明確な逸脱

[intent] 候補と「実際に話者が続けた内容」の意図一致:
2: ほぼ同じ内容・意図
1: 表現は違うが方向性・話題は同じ
0: 別の内容

説明は不要。次の形式のJSONのみを出力する: {"coherence": 0, "intent": 0}
/no_think"""

JUDGE_SYSTEM_EN = """You are an evaluator of speech-continuation candidates. When a speaker stalls mid-utterance, the system speaks a short continuation on their behalf. Evaluate the candidate.

[coherence] Does the candidate follow the flow of the context so far?
2: consistent with the topic, tone, and what was already said; a listener would not find it odd
1: no major break, but somewhat abrupt, unnatural, or off
0: clear deviation — topic drift, contradiction with what was said, nonsense, or sounding like a reply to the speaker instead of their own continuation

[intent] Does the candidate match what the speaker actually said next?
2: nearly the same content/intent
1: different wording but same direction/topic
0: different content

No explanations. Output ONLY JSON in this form: {"coherence": 0, "intent": 0}
/no_think"""


def build_user(history: str, partial: str, pred: str, truth: str, lang: str = "ja") -> str:
    ctx = (history + "\n" if history else "") + partial
    if lang == "en":
        return (
            f"<context so far>\n{ctx}\n</context so far>\n"
            f"<continuation candidate>\n{pred}\n</continuation candidate>\n"
            f"<what the speaker actually said next>\n{truth[:80]}\n</what the speaker actually said next>\n"
            "Evaluate as JSON:"
        )
    return (
        f"<これまでの文脈>\n{ctx}\n</これまでの文脈>\n"
        f"<継続候補>\n{pred}\n</継続候補>\n"
        f"<実際の続き>\n{truth[:40]}\n</実際の続き>\n"
        "JSONで評価:"
    )


_JSON_RE = re.compile(r'"coherence"\s*:\s*([0-2])\s*,\s*"intent"\s*:\s*([0-2])')
_JSON_RE2 = re.compile(r'"intent"\s*:\s*([0-2])\s*,\s*"coherence"\s*:\s*([0-2])')


def parse_judgement(text: str) -> tuple[int, int] | None:
    text = text.split("</think>")[-1]
    m = _JSON_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _JSON_RE2.search(text)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None


async def judge_row(client: httpx.AsyncClient, url: str, row: dict, history: str,
                    sem: asyncio.Semaphore, lang: str = "ja") -> dict:
    async with sem:
        payload = {
            "model": "judge",
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_EN if lang == "en" else JUDGE_SYSTEM},
                {"role": "user", "content": build_user(history, row["input_latest"],
                                                       row["pred"], row["truth"], lang=lang)},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for _ in range(2):
            resp = await client.post(f"{url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            parsed = parse_judgement(content)
            if parsed:
                row["coherence"], row["intent"] = parsed
                return row
        row["coherence"], row["intent"] = None, None
        row["judge_raw"] = content[:200]
        return row


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def summarize(rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("coherence") is not None]
    n = len(judged)
    if n == 0:
        return {"judged": 0}
    coh = [r["coherence"] for r in judged]
    inte = [r["intent"] for r in judged]
    hit0 = [r for r in judged if r["match_len"] == 0]
    return {
        "judged": n,
        "逸脱率(coherence=0)": round(sum(1 for c in coh if c == 0) / n, 3),
        "違和感率(coherence<=1)": round(sum(1 for c in coh if c <= 1) / n, 3),
        "coherence平均": round(sum(coh) / n, 2),
        "意図一致率(intent>=1)": round(sum(1 for i in inte if i >= 1) / n, 3),
        "意図一致率(intent=2)": round(sum(1 for i in inte if i == 2) / n, 3),
        "hit@1=0のうちintent>=1(惜しい外れ)": round(
            sum(1 for r in hit0 if r["intent"] >= 1) / max(1, len(hit0)), 3),
        "hit@1=0のうちcoherence=2": round(
            sum(1 for r in hit0 if r["coherence"] == 2) / max(1, len(hit0)), 3),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    ap.add_argument("--fixtures", default=None, help="省略時は言語別既定")
    ap.add_argument("--judge-url", default=f"http://{config.server_host()}:18081")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.fixtures is None:
        args.fixtures = ("fixtures/en_utterances.jsonl" if args.lang == "en"
                         else "fixtures/ja_utterances.jsonl")
    fixtures = {str(f["id"]): f for f in load_jsonl(args.fixtures)}
    rows = [r for r in load_jsonl(args.rows) if r["speakable"] and r["pred"]]
    if args.limit:
        rows = rows[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            judge_row(client, args.judge_url.rstrip("/"), dict(r),
                      fixtures.get(r["fixture_id"], {}).get("history", ""), sem,
                      lang=args.lang)
            for r in rows
        ]
        judged = await asyncio.gather(*tasks)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in judged:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = summarize(list(judged))
    print(f"== judge report: {args.rows} ==")
    for k, v in report.items():
        print(f"  {k}: {v}")
    # 逸脱と判定された例を表示
    bad = [r for r in judged if r.get("coherence") == 0][:5]
    if bad:
        print("  -- 逸脱例 (coherence=0):")
        for r in bad:
            print(f"     …{r['input_latest'][-14:]} | 予測: {r['pred']} | 実際: {r['truth'][:14]}…")


if __name__ == "__main__":
    asyncio.run(main())
