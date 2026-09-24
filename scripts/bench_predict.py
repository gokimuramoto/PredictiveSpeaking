"""継続予測の単体ベンチCLI。

例:
  uv run python scripts/bench_predict.py --url http://127.0.0.1:8080 --mode raw
  uv run python scripts/bench_predict.py --url http://<A6000サーバ>:8080 --mode raw --n 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.bench.runner import format_report, run_bench  # noqa: E402
from predictive_speaking.predict.llamacpp import LlamaCppPredictor  # noqa: E402
from predictive_speaking.predict.mock import MockPredictor  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--backend", choices=["llamacpp", "mock"], default="llamacpp")
    ap.add_argument("--mode", choices=["raw", "chat"], default="raw")
    ap.add_argument("--n", type=int, default=3, help="raw modeの候補数(サーバの--parallel以下)")
    ap.add_argument("--n-predict", type=int, default=12)
    ap.add_argument("--preamble", default="")
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--dry", type=float, default=0.75)
    ap.add_argument("--n-probs", type=int, default=1,
                    help="0でlogprob信頼度を切る(ローカルiGPUでは~1s/24tokの節約)")
    ap.add_argument("--strip-history", action="store_true",
                    help="予測器に履歴を渡さない(文脈効果のアブレーション用。ジャッジは常に完全文脈で採点)")
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    ap.add_argument("--fixtures", default="fixtures/ja_utterances.jsonl")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--min-prefix", type=int, default=5)
    ap.add_argument("--out", default=None, help="行単位結果のJSONL出力先")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.backend == "mock":
        predictor = MockPredictor(delay_ms=10)
    else:
        predictor = LlamaCppPredictor(
            base_url=args.url, mode=args.mode, n_candidates=args.n,
            n_predict=args.n_predict, preamble=args.preamble,
            repeat_penalty=args.repeat_penalty, dry_multiplier=args.dry,
            lang=args.lang, n_probs=args.n_probs,
        )
    if args.lang == "en" and args.fixtures == "fixtures/ja_utterances.jsonl":
        args.fixtures = "fixtures/en_utterances.jsonl"

    if args.strip_history:
        import json as _json
        import tempfile

        with open(args.fixtures, encoding="utf-8") as fh:
            items = [{**_json.loads(x), "history": ""} for x in fh if x.strip()]
        tf = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for it in items:
            tf.write(_json.dumps(it, ensure_ascii=False) + "\n")
        tf.close()
        args.fixtures = tf.name

    try:
        report = await run_bench(
            predictor, args.fixtures,
            min_prefix=args.min_prefix, stride=args.stride,
            max_steps=args.max_steps, max_items=args.max_items,
            out_path=args.out,
        )
    finally:
        await predictor.close()

    label = args.label or f"{args.backend}:{args.mode}"
    print(format_report(label, report))
    if args.out:
        print(f"rows -> {args.out}")
    # 機械可読サマリも1行出す(結果比較用)
    print("JSON:", json.dumps({"label": label, **report}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
