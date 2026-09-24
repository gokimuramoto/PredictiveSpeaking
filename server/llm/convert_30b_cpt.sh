#!/bin/bash
# 30B-A3B CPT版のGGUF変換パイプライン(サーバ上でnohup実行される)
set -e
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}  # llama.cpp/ と models/ がある作業ディレクトリ
cd "$PS_WORK"
export PATH=$HOME/.local/bin:$PATH
PY=${PY:-python3}  # huggingface_hub・sentencepiece等が入ったPython

uv pip install -q --python $PY "huggingface_hub[hf_transfer]" sentencepiece protobuf mistral-common
mkdir -p models/hf_30b_cpt

$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("tokyotech-llm/Qwen3-Swallow-30B-A3B-CPT-v0.2", local_dir="models/hf_30b_cpt")
print("DOWNLOAD_DONE", flush=True)
EOF

$PY llama.cpp/convert_hf_to_gguf.py models/hf_30b_cpt \
  --outfile models/Qwen3-Swallow-30B-A3B-CPT-v0.2-bf16.gguf --outtype bf16

./llama.cpp/build/bin/llama-quantize \
  models/Qwen3-Swallow-30B-A3B-CPT-v0.2-bf16.gguf \
  models/Qwen3-Swallow-30B-A3B-CPT-v0.2-Q4_K_M.gguf Q4_K_M

rm -f models/Qwen3-Swallow-30B-A3B-CPT-v0.2-bf16.gguf
echo CONVERT_PIPELINE_DONE
