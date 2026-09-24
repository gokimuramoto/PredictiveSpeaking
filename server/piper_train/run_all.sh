#!/bin/bash
# ③ Piperファインチューン一括実行 (GPUサーバ)
#  1. データ合成(gen_all.sh, gen.pid)の終了を待つ
#  2. chatterbox-vllm(18084)を停止してVRAMを学習に回す(他の常駐サービスには触れない)
#  3. 声ごとに fine-tune(lessac medium から) → ONNX書き出し
#  4. 終了後 chatterbox-vllm を復旧
# 使い方: VOICES="alice bob" nohup bash run_all.sh > $PIPER_WORK/run_all.log 2>&1 &
# 出力:   $PIPER_WORK/out/<voice>/en_US-<voice>-medium.onnx(+.onnx.json) → アプリの models/piper/ へ
#
# 学習・書き出しは必ず piper_wrap.py 経由で起動する(torch 2.6+ のweights_only対策と
# ONNX書き出しの従来エクスポータ強制)。目安: batch32でVRAM~11GB、約67分/声(A6000)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPER_WORK=${PIPER_WORK:-$HOME/projects/piper_voice}
export PATH=$HOME/.local/bin:$PATH
cd "$PIPER_WORK"
PY=piper1-gpl/.venv/bin/python
CKPT=ckpt/lessac_medium_clean.ckpt   # setup_piper.sh が hyper_parameters を除去して作るもの
BASE_EPOCH=2164            # lessac medium の学習済みepoch(再開扱いなのでこれに加算する)
ADD_EPOCHS=${ADD_EPOCHS:-250}
: "${VOICES:?VOICES に学習する声IDを空白区切りで指定してください}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. データ合成待ち
while [ -f gen.pid ] && kill -0 $(cat gen.pid) 2>/dev/null; do sleep 60; done
log "dataset generation finished"
for v in $VOICES; do
  n=$(ls data/$v/wav 2>/dev/null | wc -l)
  log "dataset $v: $n wavs"
  if [ "$n" -lt 500 ]; then log "ERROR: dataset $v too small"; exit 1; fi
done

# 2. chatterbox-vllm 停止(VRAM ~9GB解放)。vLLMはエンジン子プロセスを持つため
#    親だけkillすると子が孤児化してVRAMを握り続ける → cwdで特定して全部止める
bash "$HERE/cleanup_gpu.sh" | sed 's/^/[cleanup] /'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

train_voice() {
  local v=$1 bs=$2
  local out=out/$v
  mkdir -p $out cache/$v
  log "train $v start (batch=$bs, max_epochs=$((BASE_EPOCH+ADD_EPOCHS)))"
  $PY "$HERE/piper_wrap.py" piper.train fit \
    --data.voice_name "$v" \
    --data.csv_path data/$v/metadata.csv \
    --data.audio_dir data/$v/wav \
    --model.sample_rate 22050 \
    --data.espeak_voice en-us \
    --data.cache_dir cache/$v \
    --data.config_path $out/config.json \
    --data.batch_size $bs \
    --data.validation_split 0.0 \
    --data.num_test_examples 0 \
    --ckpt_path $CKPT \
    --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 32 \
    --trainer.max_epochs $((BASE_EPOCH+ADD_EPOCHS)) \
    --trainer.log_every_n_steps 50 \
    --trainer.default_root_dir $out \
    > $out/train_bs$bs.log 2>&1
  return $?
}

# 3. 各声を学習→書き出し(OOM時はbatch16で再試行)
for v in $VOICES; do
  if train_voice $v 32; then
    log "train $v done (batch 32)"
  else
    log "train $v failed at batch 32 (tail):"; tail -5 out/$v/train_bs32.log
    if grep -qi "out of memory" out/$v/train_bs32.log; then
      log "retrying $v with batch 16"
      train_voice $v 16 && log "train $v done (batch 16)" || { log "train $v FAILED"; tail -15 out/$v/train_bs16.log; continue; }
    else
      continue
    fi
  fi
  ck=$(find out/$v -name "*.ckpt" -newer $CKPT | sort | tail -1)
  log "export $v from $ck"
  $PY "$HERE/piper_wrap.py" piper.train.export_onnx --checkpoint "$ck" --output-file out/$v/en_US-$v-medium.onnx \
    > out/$v/export.log 2>&1 && cp out/$v/config.json out/$v/en_US-$v-medium.onnx.json \
    && log "EXPORTED $v: $(ls -la out/$v/en_US-$v-medium.onnx | awk '{print $5}') bytes" \
    || { log "export $v FAILED"; tail -5 out/$v/export.log; }
done

# 4. chatterbox-vllm 復旧
bash "$HERE/../chatterbox/setup_chatterbox_vllm.sh" | tail -1
log "chatterbox-vllm relaunched"
log "ALL_DONE"
