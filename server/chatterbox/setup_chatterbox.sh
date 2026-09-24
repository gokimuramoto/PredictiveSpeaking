#!/bin/bash
# Chatterbox EN TTS (torch版, port 18083) の環境構築+起動 (GPUサーバ)
#   通常は高速な chatterbox-vllm (setup_chatterbox_vllm.sh) を使う。
#   こちらは声の参照音声(voices_en/)の置き場所を兼ねる。
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$PS_WORK/chatterbox_server"
cd "$PS_WORK/chatterbox_server"

if [ ! -d .venv ]; then
  uv venv --python 3.11
fi
# setuptools<81: perth(音声透かし)がpkg_resources必須。81以降は同梱されず
# PerthImplicitWatermarker=Noneとなりモデル初期化がTypeErrorで落ちる
uv pip install -q --python .venv/bin/python chatterbox-tts fastapi 'uvicorn[standard]' soundfile 'setuptools<81'
echo "CHATTERBOX_INSTALLED"

cp "$HERE/chatterbox_server.py" .
if [ -f srv.pid ] && kill -0 $(cat srv.pid) 2>/dev/null; then
  kill $(cat srv.pid); sleep 2
fi
PORT=18083 nohup .venv/bin/python chatterbox_server.py > srv.log 2>&1 < /dev/null &
echo $! > srv.pid

for i in $(seq 1 180); do
  if ! kill -0 $(cat srv.pid) 2>/dev/null; then echo CRASHED; tail -20 srv.log; exit 1; fi
  if curl -s http://127.0.0.1:18083/health | grep -q chatterbox; then
    echo "CHATTERBOX_READY ~$((i*5))s"
    exit 0
  fi
  sleep 5
done
echo TIMEOUT; tail -20 srv.log; exit 1
