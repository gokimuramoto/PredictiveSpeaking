"""モデルトーナメント: サーバのGGUFを順に差し替えてベンチ+ジャッジを回し、結果表を出す。

  uv run python scripts/tournament.py --models Qwen3-8B-Base.Q8_0.gguf llm-jp-3-13b-Q8_0.gguf
終了時は必ず既定モデル(8B CPT Q8)に復旧する。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os  # noqa: E402

from predictive_speaking import config  # noqa: E402

config.load_env()
# サーバへのssh接続名(~/.ssh/configのHost)とLLM/ジャッジのURLは .env で指定する
SERVER = os.environ.get("PS_SSH_HOST", "gpu-server")
LLM_URL = f"http://{config.server_host()}:18080"
JUDGE_URL = f"http://{config.server_host()}:18081"
DEFAULT_MODEL = "Qwen3-Swallow-8B-CPT-v0.2_Q8_0.gguf"

SWAP_CMD = (
    "cd ${PS_WORK:-$HOME/projects/PredictiveSpeaking}; "
    "if [ -f logs/llama.pid ] && kill -0 $(cat logs/llama.pid) 2>/dev/null; then "
    "kill $(cat logs/llama.pid); sleep 3; fi; "
    "LD_LIBRARY_PATH=llama.cpp/build/bin nohup ./llama.cpp/build/bin/llama-server "
    "-m 'models/{model}' -ngl 99 --ctx-size 8192 --parallel 3 "
    "--host 0.0.0.0 --port 18080 --no-webui > logs/llama_tournament.log 2>&1 & "
    "echo $! > logs/llama.pid"
)


def swap(model: str) -> bool:
    subprocess.run(["ssh", "-o", "BatchMode=yes", SERVER, SWAP_CMD.format(model=model)],
                   capture_output=True, timeout=60)
    for _ in range(60):
        try:
            if httpx.get(f"{LLM_URL}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_cli(args: list[str]) -> str:
    r = subprocess.run(["uv", "run", "python"] + args, capture_output=True,
                       text=True, encoding="utf-8", timeout=900)
    return r.stdout + r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    args = ap.parse_args()

    results = []
    try:
        for model in args.models:
            name = re.sub(r"\.gguf$", "", model).replace(".", "_")
            print(f"\n===== {model} =====", flush=True)
            if not swap(model):
                print("  サーバ起動失敗 — スキップ", flush=True)
                continue
            out = run_cli(["scripts/bench_predict.py", "--url", LLM_URL, "--mode", "raw",
                           "--n", "3", "--n-predict", "24", "--lang", args.lang,
                           "--out", f"logs/t_{args.lang}_{name}.jsonl", "--label", name])
            m = re.search(r'JSON: (\{.*\})', out)
            bench = json.loads(m.group(1)) if m else {}
            row = {"model": model, "hit@1": bench.get("hit@1"), "hit@2": bench.get("hit@2"),
                   "speak_rate": bench.get("speak_rate"),
                   "wall_p50": (bench.get("wall_ms") or {}).get("p50")}
            print(f"  bench: {row}", flush=True)
            if not args.skip_judge:
                jout = run_cli(["scripts/judge_bench.py", "--rows", f"logs/t_{name}.jsonl",
                                "--judge-url", JUDGE_URL, "--out", f"logs/t_{name}_judged.jsonl"])
                for key, pat in [("逸脱率", r"逸脱率\(coherence=0\): ([\d.]+)"),
                                 ("coherence", r"coherence平均: ([\d.]+)"),
                                 ("意図一致", r"意図一致率\(intent>=1\): ([\d.]+)")]:
                    mm = re.search(pat, jout)
                    row[key] = float(mm.group(1)) if mm else None
                print(f"  judge: 逸脱率={row.get('逸脱率')} 意図一致={row.get('意図一致')}", flush=True)
            results.append(row)
    finally:
        print(f"\n既定モデルへ復旧: {DEFAULT_MODEL}", flush=True)
        swap(DEFAULT_MODEL)

    print("\n===== 結果 =====")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    Path("logs/tournament_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
