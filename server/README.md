# server/ — GPUサーバ側

既定構成(英語・ローカル動作)では不要です。ここは次の用途のためのもの:

- **本人声Piperモデルの作成**(ボイスクローン合成 → fine-tune)
- **日本語版**(大きいLLMと日本語TTSをGPUサーバで常駐)
- 研究用のLLM変換・ベンチ

想定環境: Linux + NVIDIA GPU(開発時は RTX A6000 48GB)、[uv](https://docs.astral.sh/uv/)。
リポジトリをサーバの `~/projects/PredictiveSpeaking` にcloneする前提です
(作業ディレクトリは環境変数 `PS_WORK` / `PIPER_WORK` で変更可)。

## サービスとポート

| ポート | サービス | 用途 | 構築 |
|---|---|---|---|
| 18084 | Chatterbox-vLLM(英語ボイスクローンTTS, 高速版) | Piper学習データの合成 / `--tts-backend chatterbox-vllm` | `chatterbox/setup_chatterbox_vllm.sh` |
| 18083 | Chatterbox(torch版) | 同上の従来版。声の参照音声 `voices_en/` の置き場 | `chatterbox/setup_chatterbox.sh` |
| 18080 | llama-server: GPT-OSS-Swallow-20B-SFT Q8 | 日本語版の予測 | 下記 |
| 18082 | llama-server: Qwen3-8B-Base Q8(任意) | 英語をサーバで予測する場合 | 下記 |
| 18081 | llama-server: Qwen3-Swallow-8B-SFT Q4(任意) | 評価用LLMジャッジ(`judge_bench.py`) | 下記 |
| 8088 | Irodori-TTS-Server | 日本語TTS(別プロジェクト) | [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) |

アプリ側は `.env` の `PS_SERVER_HOSTS` にこのサーバのホストを書く(カンマ区切りで複数可、先に届いたものを使用)。

## llama-server(日本語版・評価用)

```bash
cd ~/projects/PredictiveSpeaking
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON && cmake --build llama.cpp/build -j --target llama-server
# models/ にGGUFを置いて起動(例: 日本語版)
LD_LIBRARY_PATH=llama.cpp/build/bin nohup ./llama.cpp/build/bin/llama-server \
  -m models/GPT-OSS-Swallow-20B-SFT-v0.1-Q8_0.gguf -ngl 99 --ctx-size 8192 --parallel 3 \
  --host 0.0.0.0 --port 18080 --no-webui > logs/llama.log 2>&1 &
```

GGUFが配布されていないモデルは `llm/convert_generic.sh <hf_repo> <出力名> [量子化]` で変換できます。

## 本人声Piperモデルの作成

実録音の代わりに、Chatterboxのボイスクローンで本人声の学習データを合成してPiperをfine-tuneします。

1. **声の登録**: Chatterbox-vLLMを起動し、本人の参照音声(静かな環境で10秒以上)を登録する
   ```bash
   bash chatterbox/setup_chatterbox_vllm.sh
   ```
   登録はアプリUIの「+声を登録」(`--tts-backend chatterbox-vllm` 起動時)か、手元のPCから:
   `uv run python scripts/smoke_chatterbox.py --url http://<server>:18084 --register my_voice.wav --voice-id alice`
2. **学習環境の構築**(初回のみ)
   ```bash
   bash piper_train/setup_piper.sh
   ```
3. **学習データの合成**(CMU ARCTIC 1132文、1声あたり約30分。中断しても再開可能)
   ```bash
   VOICES="alice" bash piper_train/gen_all.sh        # 進捗: tail -f ~/projects/piper_voice/gen.log
   ```
4. **fine-tune → ONNX書き出し**(合成完了を待って自動開始。1声あたり約67分)
   ```bash
   VOICES="alice" nohup bash piper_train/run_all.sh > ~/projects/piper_voice/run_all.log 2>&1 &
   ```
5. 出力 `~/projects/piper_voice/out/alice/en_US-alice-medium.onnx` と `.onnx.json` を
   手元PCの `models/piper/` にコピーする

### 構築時の罠(スクリプトで対処済み)

| 症状 | 原因と対処 |
|---|---|
| Chatterbox起動時 `TypeError` | 透かし(perth)が`pkg_resources`必須 → `setuptools<81` 固定 |
| chatterbox-vllm の EngineCore 起動失敗 | PyPI版は古い → リポジトリをclone。`chatterbox_vllm`のimportはトップレベル必須(vLLMのspawn子プロセス対策) |
| 学習でOOM(VRAMが空かない) | vLLMの子プロセスが孤児化してVRAMを保持 → `cleanup_gpu.sh`、起動は`setsid`でプロセスグループごと停止 |
| `build_ext` 失敗 | `scikit-build` が必要(公式手順に記載なし) |
| `UnpicklingError: PosixPath` | torch 2.6+ の`weights_only` → `piper_wrap.py` 経由で起動 |
| `Parsing of ckpt_path hyperparameters failed` | 旧ckptの`hyper_parameters` → `sanitize_ckpt.py` で除去(setup_piper.shが実施) |
| ONNX書き出し失敗 | `onnxscript`不足+新エクスポータ(dynamo)非対応 → `piper_wrap.py` が従来エクスポータを強制 |

## ファイル

```
chatterbox/
  chatterbox_server.py         torch版サーバ(Irodori互換API)
  chatterbox_vllm_server.py    vLLM版サーバ(同API)
  setup_chatterbox*.sh         環境構築+起動
piper_train/
  setup_piper.sh               ① 学習環境+事前学習チェックポイント
  gen_all.sh / gen_dataset.py  ② 学習データ合成
  run_all.sh                   ③ fine-tune → ONNX書き出し → chatterbox-vllm復旧
  piper_wrap.py / sanitize_ckpt.py / cleanup_gpu.sh   上記の罠の対処
llm/
  convert_generic.sh           HFモデル → GGUF変換・量子化
  convert_30b_cpt.sh / tournament_dl.sh   モデル比較実験用
```
