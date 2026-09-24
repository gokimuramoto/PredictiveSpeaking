#!/bin/bash
# 汎用: HFモデルをダウンロード→bf16変換→量子化
# 使い方: bash convert_generic.sh <hf_repo> <出力名> [量子化タイプ]
set -e
REPO="$1"
NAME="$2"
QTYPE="${3:-Q8_0}"
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}  # llama.cpp/ と models/ がある作業ディレクトリ
cd "$PS_WORK"
export PATH=$HOME/.local/bin:$PATH
PY=${PY:-python3}  # huggingface_hub・sentencepiece等が入ったPython

mkdir -p "models/hf_$NAME"
$PY - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$REPO", local_dir="models/hf_$NAME")
print("DOWNLOAD_DONE", flush=True)
EOF

$PY llama.cpp/convert_hf_to_gguf.py "models/hf_$NAME" \
  --outfile "models/$NAME-bf16.gguf" --outtype bf16

./llama.cpp/build/bin/llama-quantize \
  "models/$NAME-bf16.gguf" "models/$NAME-$QTYPE.gguf" "$QTYPE"

rm -f "models/$NAME-bf16.gguf"
echo "CONVERT_DONE models/$NAME-$QTYPE.gguf"
