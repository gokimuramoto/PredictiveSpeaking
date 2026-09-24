"""PPL-gapスコア: 「局所的には流暢だが長い文脈に照らすと浮いている」候補の機械的検出。

gap = [logP(候補 | 全文脈) - logP(候補 | 直近の短い末尾のみ)] / 候補トークン数

gapが負に大きい = 長い文脈を考慮すると尤度が下がる = 文脈から浮いている疑い。
採点モデルは小型のbaseモデル(既定: Qwen/Qwen3-0.6B-Base, CPU実行)。
測定器としてだけでなく、将来オンラインのゲートシグナルにも同じ計算を使う。

例:
  uv run --group eval python scripts/ppl_gap.py --rows logs/bench_raw_swallow8b_q8_n3.jsonl \
    --out logs/pplgap_swallow8b.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


class Scorer:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B-Base", device: str = "cpu"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32).to(device).eval()
        self.device = device

    @torch.no_grad()
    def logprob_of(self, context: str, continuation: str) -> tuple[float, int]:
        """P(continuation | context) の合計logprobとトークン数。"""
        ctx_ids = self.tok(context, return_tensors="pt").input_ids
        full_ids = self.tok(context + continuation, return_tensors="pt").input_ids
        n_ctx = ctx_ids.shape[1]
        n_cont = full_ids.shape[1] - n_ctx
        if n_cont <= 0:
            return 0.0, 0
        out = self.model(full_ids.to(self.device))
        logprobs = torch.log_softmax(out.logits[0], dim=-1)
        total = 0.0
        for i in range(n_ctx, full_ids.shape[1]):
            total += logprobs[i - 1, full_ids[0, i]].item()
        return total, n_cont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--fixtures", default="fixtures/ja_utterances.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--short-tail-chars", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    fixtures = {str(f["id"]): f for f in load_jsonl(args.fixtures)}
    rows = [r for r in load_jsonl(args.rows) if r["speakable"] and r["pred"]]
    if args.limit:
        rows = rows[: args.limit]

    scorer = Scorer(args.model)
    results = []
    for r in rows:
        history = fixtures.get(r["fixture_id"], {}).get("history", "")
        full_ctx = (history + "\n" if history else "") + r["input_latest"]
        short_ctx = r["input_latest"][-args.short_tail_chars:]
        lp_full, n1 = scorer.logprob_of(full_ctx, r["pred"])
        lp_short, n2 = scorer.logprob_of(short_ctx, r["pred"])
        n = max(1, n1)
        r["pplgap"] = round((lp_full - lp_short) / n, 3)
        r["lp_full_per_tok"] = round(lp_full / n, 3)
        results.append(r)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    gaps = sorted(results, key=lambda r: r["pplgap"])
    vals = [r["pplgap"] for r in results]
    print(f"== ppl-gap: {args.rows} (n={len(vals)}, model={args.model}) ==")
    print(f"  mean={sum(vals)/len(vals):.3f}  min={vals and min(vals):.3f}  max={max(vals):.3f}")
    neg = sum(1 for v in vals if v < -0.5)
    print(f"  gap < -0.5 (文脈で尤度が下がる=浮き疑い): {neg}/{len(vals)} ({neg/len(vals):.1%})")
    print("  -- gap最小(浮き疑い上位):")
    for r in gaps[:5]:
        print(f"     gap={r['pplgap']:+.2f} …{r['input_latest'][-12:]} | 予測: {r['pred']}")
    print("  -- gap最大(文脈が支えている上位):")
    for r in gaps[-5:]:
        print(f"     gap={r['pplgap']:+.2f} …{r['input_latest'][-12:]} | 予測: {r['pred']}")


if __name__ == "__main__":
    main()
