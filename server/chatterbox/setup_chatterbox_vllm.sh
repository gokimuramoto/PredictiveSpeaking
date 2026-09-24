#!/bin/bash
# Chatterbox-vLLM (高速版EN TTS, port 18084) の環境構築+起動 (GPUサーバ)
#   ボイスクローン合成(Piper学習データの生成元)。声は torch版と voices_en/ を共有する。
#
# 罠メモ:
# - PyPI版(chatterbox-vllm)は古く vllm 0.10 と非互換 → リポジトリをcloneしてuv syncする
# - サーバスクリプトは chatterbox_vllm のimportを必ずトップレベルに置く
#   (vLLMのspawn子が__main__を再importする際のEnTokenizer登録に依存)
# - GPU_FRACTIONは共有GPUのため控えめに(0.15)
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$PS_WORK/chatterbox_vllm" "$PS_WORK/chatterbox_server/voices_en"
cd "$PS_WORK/chatterbox_vllm"
if [ ! -d repo ]; then
  git clone -q https://github.com/randombk/chatterbox-vllm.git repo
fi
cd repo
uv sync -q
uv pip install -q --python .venv/bin/python fastapi 'uvicorn[standard]' soundfile 'setuptools<81'
cp "$HERE/chatterbox_vllm_server.py" .

# vLLMはエンジン子プロセスを持つ: 親だけkillすると子が孤児化してVRAMを握り続けるため、
# setsidで独立プロセスグループにして kill -- -PGID で子ごと止める
if [ -f ../srv.pid ] && kill -0 $(cat ../srv.pid) 2>/dev/null; then
  kill -- -$(cat ../srv.pid) 2>/dev/null || kill $(cat ../srv.pid); sleep 3
fi
PORT=18084 GPU_FRACTION=${GPU_FRACTION:-0.15} VOICES_DIR="$PS_WORK/chatterbox_server/voices_en" \
  setsid .venv/bin/python chatterbox_vllm_server.py > ../srv.log 2>&1 < /dev/null &
echo $! > ../srv.pid
for i in $(seq 1 90); do
  if ! kill -0 $(cat ../srv.pid) 2>/dev/null; then echo CRASHED; tail -20 ../srv.log; exit 1; fi
  if curl -s -m 2 http://127.0.0.1:18084/health | grep -q vllm; then
    echo "CBX_VLLM_READY"
    exit 0
  fi
  sleep 5
done
echo TIMEOUT; tail -20 ../srv.log; exit 1
