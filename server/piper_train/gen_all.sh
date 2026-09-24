#!/bin/bash
# ② 学習データの合成: CMU ARCTIC(音素バランス英文1132本, パブリックドメイン)を
#    chatterbox-vllm に登録済みのクローン声で合成する(バックグラウンド・中断しても再開可能)
#
# 前提: chatterbox-vllm (port 18084) が起動済みで、VOICES の声が登録済みであること
#       (server/chatterbox/setup_chatterbox_vllm.sh / アプリUIの「+声を登録」)
# 使い方: VOICES="alice bob" bash gen_all.sh     → 進捗: tail -f $PIPER_WORK/gen.log
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PS_WORK=${PS_WORK:-$HOME/projects/PredictiveSpeaking}
PIPER_WORK=${PIPER_WORK:-$HOME/projects/piper_voice}
: "${VOICES:?VOICES に合成する声IDを空白区切りで指定してください (例: VOICES=\"alice bob\")}"

mkdir -p "$PIPER_WORK/data"
cd "$PIPER_WORK"
if [ ! -f cmuarctic.data ]; then
  curl -sfL -o cmuarctic.data "http://festvox.org/cmu_arctic/cmuarctic.data"
fi
echo "prompts: $(grep -c '^( ' cmuarctic.data)"

# httpx / soundfile / torchaudio が入っている chatterbox-vllm の環境を流用する
PY=${PY:-$PS_WORK/chatterbox_vllm/repo/.venv/bin/python}
cmds=()
for v in $VOICES; do
  cmds+=("$PY $HERE/gen_dataset.py --voice $v --out data/$v")
done
joined=$(printf " && %s" "${cmds[@]}")
nohup bash -c "${joined# && }" > gen.log 2>&1 < /dev/null &
echo $! > gen.pid
echo "GEN_STARTED pid=$(cat gen.pid)  (目安: 1声あたり約30分)"
