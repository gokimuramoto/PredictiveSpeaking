# PredictiveSpeaking

話している最中に**次の発話を常に予測**し、言葉に詰まったら**本人の声で続きを補う**システム。

予測は止めずに回し続け、「いつ補助音声を出すか」だけをゲート(ボタン/自動)で制御する設計です。
候補の音声は先行合成してキャッシュしておくので、補助は押した瞬間(または詰まって約0.8秒後)に鳴ります。

```
マイク ─▶ ASR(Azure) ─▶ 常時予測(ローカルLLM) ─▶ 先行合成(Piper, 本人声) ─▶ ゲート ─▶ スピーカー
                              ▲ KVキャッシュ常駐で差分だけ計算            ▲ タップ or 自動(無音検出)
```

## 既定構成

| 要素 | 実装 | 動作場所 | 実測 |
|---|---|---|---|
| 音声認識 | Azure Speech (en-US) | クラウド | partial更新 300〜500ms |
| 発話予測 | Qwen3-4B (llama.cpp, 自動起動) | このPC | 1予測 約0.5秒 |
| 音声合成 | Piper (本人声でfine-tune済みVITS) | このPC(CPU) | 1文 70〜110ms |

音声認識以外はローカルで完結します(動作確認: Windows 11 / Radeon 8060S iGPU)。

## クイックスタート

必要なもの: [uv](https://docs.astral.sh/uv/)、Azure Speech のキー、マイクとスピーカー(ヘッドセット推奨)。

```powershell
uv sync                                      # 依存関係のインストール
copy .env.example .env                       # .env を作り AZURE_SPEECH_KEY を記入
uv run python scripts/setup_models.py        # LLM・llama.cpp・市販Piper声を models/ に取得(約2.6GB)
uv run python scripts/run_app.py             # 起動 → http://127.0.0.1:8770 を開く
```

本人声のPiperモデル(`<name>.onnx` と `<name>.onnx.json`)があれば `models/piper/` に置くと自動で優先されます。
無ければ市販の英語声(lessac)で動きます。

## 使い方

| 操作 | 動作 |
|---|---|
| **Space / 発話ボタン** | 表示中の候補を発話(離しても最後まで再生。もう一度押すと停止) |
| **自動発話(右下トグル)** | 候補の音声が準備済みで、発話が約0.8秒止まると自動で補助。話し始めれば即中断 |
| **話題メモ(ヘッダ)** | 話す予定の内容・固有名詞を数行書く → 予測の逸脱が半減し、認識の語彙ヒントにもなる |
| **声(ヘッダ)** | Piperモデルを切替。最後に選んだ声が次回起動時の既定になる |
| **マイク(ヘッダ)** | 入力レベル表示。赤い「停止/未接続」ならマイクが届いていない |
| **リセット** | 履歴・候補・キャッシュをクリア |

英語の予測品質は**文脈量**で大きく変わります(履歴なし→400〜600字で逸脱率 52.8%→25.0%)。
話題メモを使い、セッションを切らずに話すほど予測が安定します。

## 主なオプション

```powershell
uv run python scripts/run_app.py --voice en_US-myname-medium   # 声を指定
uv run python scripts/run_app.py --mic-device realtek          # マイクを名前の一部で指定
uv run python scripts/run_app.py --context-file memo.txt       # 話題メモをファイルから
uv run python scripts/run_app.py --asr vosk                    # 完全オフライン(setup_models.py --vosk が必要。精度は低い)
uv run python scripts/run_app.py --asr typed --no-tts          # 音声なしの動作確認(fixturesを文字送り)
```

| 引数 | 既定 | 説明 |
|---|---|---|
| `--lang` | `en` | `ja` はGPUサーバ構成(下記) |
| `--asr` | `azure` | `vosk`=オフライン / `typed`=文字送りテスト |
| `--llm` | `local` | `server` で GPUサーバの llama-server を使う |
| `--tts-backend` | `piper` | `kokoro`(ローカル予備) / `chatterbox-vllm`・`chatterbox`(サーバ、即時クローン) / `irodori`(日本語) |

`uv run python scripts/run_app.py --help` で全オプションを確認できます。

### 日本語版・GPUサーバ構成

日本語はローカルで十分な品質のLLM/TTSがないため、GPUサーバ上の常駐サービスを使います
(LLM: GPT-OSS-Swallow-20B :18080 / TTS: Irodori-TTS :8088)。
`.env` の `PS_SERVER_HOSTS` にサーバのホストを優先順に書き、`--lang ja` で起動します。
サーバ側の構築は [server/README.md](server/README.md) を参照。

## 本人声モデルの作り方

実録音の代わりに、ボイスクローンTTS(Chatterbox)で本人声の学習データ約1100文を合成し、
Piperをfine-tuneします(GPUサーバで1声あたり合成約30分+学習約67分)。
手順は [server/README.md](server/README.md#本人声piperモデルの作成) にあります。

本人声モデルは実在人物の声なので**このリポジトリには含めません**(`models/` はgit管理外)。

## 開発

```powershell
uv run pytest                                          # 単体テスト
uv run python scripts/run_app.py --asr typed --port 8771
uv run python scripts/smoke_e2e.py --part 2 --app-url http://127.0.0.1:8771   # タップ発話E2E
uv run python scripts/smoke_e2e.py --part 3 --app-url http://127.0.0.1:8771   # 自動発話E2E
uv run python scripts/smoke_piper.py                   # Piperの速度計測+サンプルwav
```

評価系(予測精度・逸脱率):

- `scripts/bench_predict.py` — 予測のhit@k・速度(fixturesの発話を1語ずつ伸ばして予測)
- `scripts/judge_bench.py` — LLMジャッジによる逸脱率(coherence)・意図一致の採点
- `scripts/analyze_session.py` — 実セッションログ(`logs/session_*.jsonl`)の分析
- `scripts/bakeoff_asr.py` / `scripts/record_wav.py` — 実声でのASR比較

## 構成

```
scripts/run_app.py        アプリ本体(起動スクリプト)
scripts/setup_models.py   モデル・llama.cppの取得
src/predictive_speaking/
  asr/        Azure / Vosk / 文字送り
  predict/    llama-serverクライアント(KV常駐)・常時予測ループ
  tts/        Piper / Kokoro / サーバ型(Irodori・Chatterbox)クライアント、先行合成キャッシュ
  text/       日本語・英語のテキスト処理(フィラー除去・自然な切断・助詞縫合)
  validate.py 候補の検証(退化・メタ発話・長さ)
  session.py  ASR→予測の結線 / webapp.py  UI・ゲート・自動発話の制御
  config.py   環境依存の設定(.env)
web/          UI(index.html)
server/       GPUサーバ側(Chatterbox、Piper学習パイプライン、LLM変換)
tests/        単体テスト
fixtures/     ベンチ用の発話データ(日本語・英語)
docs/         設計書・実験記録
models/       モデル置き場(git管理外)
```

## ドキュメント

- [docs/BENCH_RESULTS.md](docs/BENCH_RESULTS.md) — 実験の時系列記録(モデル選定・ASR比較・遅延・逸脱率)
- [docs/ALWAYS_ON_DESIGN_2026-08.md](docs/ALWAYS_ON_DESIGN_2026-08.md) — 常時予測+ゲート分離の設計
- [docs/REDESIGN_PROPOSAL_2026-07.md](docs/REDESIGN_PROPOSAL_2026-07.md) — 部品選定の調査
- [docs/PredictiveSpeaking_handoff.md](docs/PredictiveSpeaking_handoff.md) — 旧版(Node.js)からの引き継ぎ仕様
