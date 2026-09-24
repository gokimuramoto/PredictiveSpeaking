#!/bin/bash
# モデルトーナメント用GGUFの探索+ダウンロード
# 各リポジトリのファイル一覧をHF APIで取得し、Q8_0 > Q6_K > Q4_K_M の優先で1つ選んで落とす
set -u
cd "${PS_WORK:-$HOME/projects/PredictiveSpeaking}/models"

REPOS="mradermacher/Qwen3-8B-Base-GGUF mradermacher/Qwen3-14B-Base-GGUF mmnga/llm-jp-3-13b-gguf mmnga/sarashina2.2-3b-gguf"

for repo in $REPOS; do
  files=$(curl -s "https://huggingface.co/api/models/$repo" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit()
names = [s["rfilename"] for s in d.get("siblings", []) if s["rfilename"].endswith(".gguf")]
for pref in ("Q8_0", "Q6_K", "Q4_K_M"):
    cand = [n for n in names if pref in n and "of-" not in n]
    if cand:
        print(cand[0])
        break
')
  if [ -z "$files" ]; then
    echo "NOTFOUND $repo"
    continue
  fi
  out=$(basename "$files")
  if [ -f "$out" ]; then
    echo "EXISTS $out"
    continue
  fi
  echo "DL $repo/$files"
  curl -sfL -o "$out" "https://huggingface.co/$repo/resolve/main/$files" \
    && echo "OK $out $(stat -c %s "$out")" || echo "FAIL $repo/$files"
done
echo TOURNAMENT_DL_DONE
