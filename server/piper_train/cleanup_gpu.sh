#!/bin/bash
# 学習前のVRAM整理: このプロジェクトの chatterbox-vllm だけを止める(他の常駐サービスには触れない)
#   vLLMはエンジン子プロセスを持ち、親だけkillすると子が孤児化してVRAMを握り続ける。
#   そこで cwd が chatterbox_vllm 配下のGPUプロセスを特定して全部止める。
set -u
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}
MINE=$PS_WORK/chatterbox_vllm
for sig in TERM KILL; do
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    cwd=$(readlink /proc/$p/cwd 2>/dev/null || true)
    case "$cwd" in
      $MINE*) echo "kill -$sig $p (cwd=$cwd)"; kill -$sig $p 2>/dev/null || true ;;
    esac
  done
  sleep 3
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
