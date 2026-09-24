#!/bin/bash
# ① piper1-gpl 学習環境の構築 + 事前学習チェックポイントの準備 (GPUサーバ)
#   docs: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md
#
# 罠メモ(対処済み):
#   - setup.py build_ext には scikit-build が必要(公式手順に記載なし)
#   - 旧rhasspy版ckptの hyper_parameters(sample_bytes等)を LightningCLI が再解析して落ちる
#     → sanitize_ckpt.py で hyper_parameters を除去したものを使う
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPER_WORK=${PIPER_WORK:-$HOME/projects/piper_voice}
export PATH=$HOME/.local/bin:$PATH

mkdir -p "$PIPER_WORK"
cd "$PIPER_WORK"
if [ ! -d piper1-gpl ]; then
  git clone -q https://github.com/OHF-voice/piper1-gpl.git
fi
cd piper1-gpl
if [ ! -d .venv ]; then
  uv venv -q --python 3.11
fi
uv pip install -q --python .venv/bin/python -e '.[train]' scikit-build cmake ninja cython onnx onnxscript
# monotonic align / espeak bridge (C拡張) のビルド
source .venv/bin/activate
./build_monotonic_align.sh 2>&1 | tail -2
python3 setup.py build_ext --inplace 2>&1 | tail -2
deactivate
.venv/bin/python -c "from piper.train.vits import monotonic_align; print('piper train env OK')"

# 事前学習チェックポイント(en_US lessac medium) → hyper_parameters除去版を作る
mkdir -p "$PIPER_WORK/ckpt"
cd "$PIPER_WORK"
if [ ! -f ckpt/lessac_medium.ckpt ]; then
  curl -sfL -o ckpt/lessac_medium.ckpt \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
fi
if [ ! -f ckpt/lessac_medium_clean.ckpt ]; then
  piper1-gpl/.venv/bin/python "$HERE/sanitize_ckpt.py" ckpt/lessac_medium.ckpt ckpt/lessac_medium_clean.ckpt
fi
ls -la ckpt/
echo "PIPER_SETUP_DONE"
