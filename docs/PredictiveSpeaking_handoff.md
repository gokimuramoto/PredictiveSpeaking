# PredictiveSpeaking プロトタイプ再実装 引き継ぎ資料

## 0. この資料の目的

本資料は、**PredictiveSpeaking** のプロトタイプをCoding Agentへ引き継ぎ、実装を開始できる状態にするための仕様・設計メモである。

PredictiveSpeakingは、ユーザーが発話中に言葉へ詰まった際、ユーザーが次に言おうとしていた短い続きを予測し、低遅延で本人に近い声として発話するシステムである。

今回の再実装では、Speech-to-Speechモデルを直接ファインチューニングするのではなく、以下の構成を採用する。

```text
ストリーミングASR
    ↓
Azure OpenAIによる発話継続予測
    ↓
日本語ストリーミングTTS
    ↓
Beatriceによる本人声へのVoice Conversion
    ↓
再生
```

重要な設計方針は、**ユーザーが詰まってから全処理を開始するのではなく、通常発話中から次発話候補を先読みし、候補音声まで準備しておくこと**である。

---

# 1. 背景

## 1.1 従来構成の問題

従来は、おおむね以下の直列処理を行っていた。

```text
詰まり検出
→ 音声認識
→ LLM推論
→ 音声合成
→ 再生
```

この構成では、各処理が順番に実行されるため、以下が累積する。

- ASR確定待ち
- LLMのネットワーク遅延
- LLMのTime To First Token
- LLMの生成時間
- TTSの初回音声生成時間
- Voice Conversionの処理時間
- 音声再生バッファリング

PredictiveSpeakingでは、ユーザーが言葉に詰まった直後に自然に補助発話を開始する必要があり、総遅延が大きな問題になる。

## 1.2 Speech-to-Speechモデルの検討

Sakana AIのKAMEやLLM-jp-Moshi-v1なども候補として検討した。

ただし、現時点では以下の理由から主構成には採用しない。

- 通常のシステムプロンプトによる制御が難しい
- 対話応答ではなく「同一話者の発話継続」という特殊タスクである
- 日本語・タスク適応にはファインチューニングが必要になる可能性が高い
- 学習コスト、GPUリソース、データ作成コストが大きい
- 誤った続きを本人の声で話すリスクが高い

KAMEからは、モデルそのものではなく、**音声処理と意味処理を並列実行し、ユーザーの発話終了前に候補を作る考え方**を採用する。

---

# 2. 今回の採用アーキテクチャ

## 2.1 推奨構成

```text
[Microphone]
    ↓
[Audio Capture / Ring Buffer]
    ├───────────────────────────────┐
    ↓                               │
[Streaming ASR]                     │
    ↓ partial transcript            │
[Transcript Stabilizer]             │
    ↓ stable prefix                 │
[Prediction Scheduler]              │
    ↓                               │
[Azure OpenAI]                      │
    ↓ short continuation            │
[Candidate Validator]               │
    ↓ approved candidate             │
[Japanese Streaming TTS]            │
    ↓ prepared PCM buffer            │
[Candidate Audio Cache]             │
                                    │
[Hesitation Detector] ←─────────────┘
    ↓ trigger
[Playback Gate]
    ↓
[Beatrice Voice Conversion]
    ↓
[Speaker Output]
```

## 2.2 基本思想

ユーザーの発話中に以下を常時行う。

1. ストリーミングASRで途中文字列を更新する
2. 安定した文字列が一定量増えたらAzure OpenAIへ送る
3. 「同じ話者が次に言いそうな短い続きを」予測する
4. 採用可能な候補であればTTSを先行実行する
5. 生成音声を一時キャッシュする
6. 実際に詰まりが検出された場合のみ再生する
7. ユーザーが自力で発話を再開したら即停止する

つまり、詰まり検出時に必要なのは、理想的には**キャッシュ済み音声を解放する処理だけ**である。

---

# 3. 採用予定モデル・サービス

## 3.1 ASR

第一候補：

```text
Qwen/Qwen3-ASR-0.6B
```

用途：

- 日本語音声のストリーミング認識
- partial transcriptの取得
- ユーザー発話中の継続的な更新

期待する特徴：

- 日本語対応
- 軽量
- 低遅延
- ストリーミング推論対応
- ローカル実行可能

注意：

- 実装時点の公式API、vLLM対応状況、必要バージョンを確認すること
- partial transcriptが過去部分を書き換える可能性を前提にする
- 最終文字起こし精度より、途中結果の安定性と更新速度を優先する

代替候補：

- Qwen3-ASR-1.7B
- ReazonSpeech系
- kotoba-whisper系
- Azure Speech to Text
- Google Cloud Speech-to-Text

初期実装では、ASRを差し替えられるインターフェースにする。

## 3.2 LLM

第一候補：

```text
Azure OpenAI
model deployment: gpt-4.1-mini 相当
```

速度比較候補：

```text
gpt-4o-mini 相当
```

重要：

- Azureでは実際のモデル名ではなく、ユーザー環境のDeployment Nameを設定から指定する
- APIバージョン、Endpoint、Keyは環境変数から読み込む
- ストリーミングレスポンスを使用する
- 出力は最大8〜12トークン程度に制限する
- reasoning系モデルは初期候補から外す
- ツール呼び出しやRAGは使わない

ローカルLLMを第一候補にしない理由：

- 軽量モデルでは日本語の発話継続精度と指示追従が弱い可能性がある
- 誤予測のコストが高い
- Azure OpenAIは発話中から先読みすればネットワーク遅延を隠せる
- ユーザーはAzure OpenAI環境をすでに利用可能

## 3.3 TTS

第一候補：

```text
Qwen3-TTS-12Hz-0.6B-CustomVoice
```

比較候補：

```text
Fun-CosyVoice3-0.5B
```

目的：

- 短い日本語テキストを低遅延で音声化
- 本人声の再現はTTSではなくBeatriceに任せる
- 固定話者または本人に近いピッチ・話速のベース音声を使用

初期プロトタイプでは、TTSモジュールも交換可能にする。

追加ベースライン：

- Style-Bert-VITS2
- Azure Speech TTS
- VOICEVOX
- 既存の低遅延日本語TTS

## 3.4 Voice Conversion

採用：

```text
Beatrice
```

役割：

- TTS音声を本人に近い声質へ変換する
- 低遅延のVoice Conversionとして利用する

注意：

- Beatrice連携方法が、仮想オーディオデバイス経由か、API/CLI/SDK経由かを最初に確認する
- 初期段階ではBeatriceを外し、TTS出力までの遅延評価を可能にする
- Beatrice込み／なしの両方をベンチマークする

---

# 4. 機能要件

## 4.1 必須機能

- マイクからリアルタイム音声を取得できる
- ASRのpartial transcriptを逐次取得できる
- ASR結果の安定部分を抽出できる
- 発話途中にAzure OpenAIへ継続候補を問い合わせられる
- 古いLLMリクエストをキャンセルできる
- LLM出力を短い継続候補として検証できる
- TTSを事前実行できる
- 候補音声を一時キャッシュできる
- 言い淀み・発話停止を検出できる
- 詰まり時だけ候補音声を再生できる
- ユーザーが発話を再開したら補助音声を停止できる
- 各処理の遅延を記録できる
- セッションログを保存できる

## 4.2 任意機能

- 複数候補の生成
- シーン別プロンプト
- 発表原稿や話題メモの入力
- 手動トリガー
- 候補をGUIに表示して選択
- 候補の採用／拒否フィードバック
- ユーザーごとの話し方設定
- ローカルLLMとの比較
- Azure OpenAIモデル間比較
- 複数TTS比較
- 誤発話防止レベルの設定

---

# 5. 非機能要件

## 5.1 レイテンシー

重要指標：

- ASR partial更新間隔
- 安定prefix生成までの時間
- LLM Time To First Token
- LLM候補確定時間
- TTS Time To First Audio
- TTS全文生成時間
- Beatrice処理時間
- 詰まり検出から補助音声再生までの時間
- ユーザー発話再開から補助音声停止までの時間

初期目標値：

```text
詰まり検出 → 補助音声開始
p50: 150 ms以下
p95: 350 ms以下
```

ただし、これは候補音声の先行生成に成功している場合の目標である。

候補が未生成の場合は、無理に遅れて話すより沈黙を優先する。

## 5.2 安全性

PredictiveSpeakingでは、誤った続きを本人の声で話すことが大きな問題になる。

以下の場合は原則として発話しない。

- 候補が複数に大きく分散する
- 文脈が不足している
- 数字を含む
- 固有名詞を含む
- 否定表現を含む
- 医療、法律、金銭、契約に関わる
- ユーザー発話が文末として完結している
- 単なる息継ぎの可能性が高い
- ユーザーがすでに発話を再開した
- 候補の生成が締切に間に合わなかった

基本原則：

```text
誤って話すくらいなら、話さない
```

---

# 6. 状態遷移

推奨状態：

```text
IDLE
LISTENING
PREDICTING
CANDIDATE_READY
HESITATION_PENDING
PLAYING_ASSIST
USER_RESUMED
COOLDOWN
ERROR
```

## 6.1 状態説明

### IDLE

- マイク待機前
- セッション未開始

### LISTENING

- ユーザーの通常発話を取得
- ASR partialを更新
- 安定prefixを監視

### PREDICTING

- LLMへ候補問い合わせ中
- 新しいstable prefixが来た場合、古い要求はキャンセル可能

### CANDIDATE_READY

- LLM候補が検証を通過
- TTS音声がキャッシュ済み
- 詰まり発生を待つ

### HESITATION_PENDING

- 一定時間の無音またはフィラーを検出
- 誤検出を防ぐため短い猶予を置く
- キャッシュ済み候補の鮮度を確認

### PLAYING_ASSIST

- Beatrice経由で候補音声を再生
- ユーザー音声の再開を監視
- 再開時は即停止

### USER_RESUMED

- ユーザーが自力で発話再開
- 補助音声停止
- 古い候補を破棄

### COOLDOWN

- 同一箇所で連続発火しないための待機
- 例：500〜1000ms

### ERROR

- モデル停止
- API失敗
- 音声デバイス切断
- タイムアウト

---

# 7. ASR安定化

## 7.1 問題

ストリーミングASRは、以下のように過去部分を書き換える。

```text
今回の研究では音声を
今回の研究では音声認識を
今回の研究では音声認識と生成を
```

毎回LLMへ送ると、無駄なAPI呼び出しと古い候補が増える。

## 7.2 Stable Prefix

直近N回のASR結果の最長共通prefixを安定部分とする。

例：

```text
R1: このシステムではユーザーが言葉に
R2: このシステムではユーザーが言葉に詰まった
R3: このシステムではユーザーが言葉に詰まったとき
```

安定prefix候補：

```text
このシステムではユーザーが言葉に
```

推奨ルール：

- 直近2〜3回に共通するprefixを採用
- 前回送信時から3〜5文字以上増加したらLLM更新
- または前回送信から300〜500ms経過したら更新
- 助詞、読点、文節境界を優先
- ASRが大きく巻き戻った場合は候補を破棄

## 7.3 ASRイベント例

```json
{
  "timestamp_ms": 1842,
  "partial_text": "このシステムではユーザーが言葉に詰まった",
  "stable_text": "このシステムではユーザーが言葉に",
  "is_final": false,
  "confidence": null
}
```

---

# 8. LLM継続予測

## 8.1 システムプロンプト案

```text
あなたは対話相手ではなく、話者本人の発話継続予測器です。

入力には、同じ話者が現在話している途中の文字起こしが与えられます。
相手への返答ではなく、同じ話者が次に発する可能性が高い短い続きを予測してください。

規則:
- 出力は発話の続きだけにする
- 1〜3文節、最大12文字程度にする
- 質問、相槌、返答を生成しない
- すでに発話された部分を繰り返さない
- 文を必要以上に完結させない
- 数字、固有名詞、否定表現は確信が高い場合だけ生成する
- 複数の続きが同程度に考えられる場合は <SILENCE> と出力する
- 入力が文末として自然な場合も <SILENCE> と出力する
- 説明、引用符、理由、確信度は出力しない
```

## 8.2 ユーザープロンプト案

```text
現在の場面:
{scene_description}

現在の話題:
{topic}

予定されている要点:
{bullet_points}

直前までの発話:
{recent_context}

現在の発話途中:
{stable_prefix}
```

## 8.3 出力例

入力：

```text
現在の発話途中:
このシステムではユーザーが言葉に詰まったとき
```

出力：

```text
次の言葉を予測して
```

または：

```text
<SILENCE>
```

## 8.4 推奨生成設定

初期値：

```text
temperature: 0.0〜0.2
max_output_tokens: 8〜12
top_p: 0.8〜1.0
stream: true
n: 1
```

Azure SDKの具体的なパラメータ名は利用APIに合わせること。

## 8.5 キャンセル戦略

- 各要求に連番`request_id`を付与する
- 新しいstable prefixが来たら、前要求をキャンセルする
- キャンセル不能な場合も、古い`request_id`の応答は破棄する
- TTS生成中でも候補が陳腐化したら破棄する

例：

```text
request 21: 「このシステムでは」
request 22: 「このシステムではユーザーが」
request 23: 「このシステムではユーザーが言葉に」
```

request 23以外の結果は採用しない。

---

# 9. 候補検証

LLM出力をそのまま発話してはいけない。

## 9.1 最低限の検証

- 空文字ではない
- `<SILENCE>`ではない
- 最大文字数以内
- 入力末尾の単純な繰り返しではない
- 改行や説明文を含まない
- 疑問文ではない
- 会話相手への返答になっていない
- 数字・固有名詞・否定語を含む場合は拒否または厳格化
- 禁止語を含まない
- 直近候補から大きく変化していない

## 9.2 候補鮮度

候補には以下を持たせる。

```json
{
  "candidate_id": "cand-000123",
  "request_id": 23,
  "source_stable_text": "このシステムではユーザーが言葉に",
  "continuation": "詰まったときに",
  "created_at_ms": 8120,
  "tts_ready_at_ms": 8270,
  "expires_at_ms": 9400
}
```

候補は古くなったら破棄する。

初期目安：

```text
有効期限: 1.0〜2.0秒
```

---

# 10. TTS先行生成

## 10.1 基本方針

LLM候補が検証を通過した時点でTTSを開始する。

詰まりを待たない。

```text
LLM候補確定
→ TTS生成
→ PCMキャッシュ
→ 詰まり待ち
```

## 10.2 音声キャッシュ

候補ごとに以下を保存する。

```json
{
  "candidate_id": "cand-000123",
  "sample_rate": 24000,
  "channels": 1,
  "pcm_format": "float32",
  "audio_duration_ms": 620,
  "audio_path": null,
  "audio_in_memory": true
}
```

MVPではメモリ上に保持する。

## 10.3 TTSキャンセル

- 新候補が来たら旧TTSをキャンセル
- キャンセルできない場合は出力だけ破棄
- 生成済み音声が古いstable prefix由来なら再生しない

---

# 11. 言い淀み検出

## 11.1 検出候補

以下を組み合わせる。

- VADによる無音長
- 音量低下
- ピッチ停止
- フィラー認識
- ASR更新停止
- 直前文が未完結
- ユーザー固有の平均ポーズ時間

## 11.2 初期ルール

単純なMVPでは以下から開始する。

```text
条件A:
ユーザーが発話中だった

条件B:
直前のstable prefixが文として未完結

条件C:
無音が350〜600ms継続

条件D:
有効な候補音声が準備済み

条件E:
直近の補助発話からcooldownを経過
```

すべて満たした場合に`HESITATION_PENDING`へ移行する。

さらに100〜200ms待ち、ユーザーが再開しなければ再生する。

## 11.3 フィラー

以下は言い淀みの強い手掛かりになる。

```text
えっと
あの
その
なんというか
えー
うーん
```

ただし、フィラーの直後にユーザーが自力で続ける場合も多いため、即時発火しない。

---

# 12. 再生・割り込み制御

## 12.1 再生条件

- 候補が有効
- 候補音声が準備済み
- ユーザーが発話していない
- cooldown中ではない
- ASRが大きく修正されていない

## 12.2 ユーザー再開時

補助発話中にユーザーが話し始めたら即停止する。

目標：

```text
ユーザー発話再開 → 補助音声停止
100ms以下
```

停止方法：

- 音声再生バッファをクリア
- Beatrice入力を停止
- Candidateを破棄
- `USER_RESUMED`へ遷移
- 一定時間`COOLDOWN`

## 12.3 同時発話

初期プロトタイプでは、AIとユーザーの長時間オーバーラップを避ける。

許容するのは、停止処理が間に合うまでの短い重なりのみ。

---

# 13. 実装インターフェース案

言語はPythonを第一候補とするが、既存システムとの都合で変更可。

## 13.1 ASR Interface

```python
class StreamingASR:
    async def start(self) -> None:
        ...

    async def push_audio(self, pcm: bytes) -> None:
        ...

    async def events(self):
        # Yield ASREvent.
        ...

    async def stop(self) -> None:
        ...
```

## 13.2 Predictor Interface

```python
class ContinuationPredictor:
    async def predict(
        self,
        stable_text: str,
        recent_context: str,
        scene_context: dict,
        request_id: int,
    ) -> "PredictionResult":
        ...

    async def cancel(self, request_id: int) -> None:
        ...
```

## 13.3 TTS Interface

```python
class StreamingTTS:
    async def synthesize(
        self,
        text: str,
        candidate_id: str,
    ) -> "AudioCandidate":
        ...

    async def cancel(self, candidate_id: str) -> None:
        ...
```

## 13.4 Voice Converter Interface

```python
class VoiceConverter:
    async def start(self) -> None:
        ...

    async def process(self, pcm: bytes) -> bytes:
        ...

    async def stop(self) -> None:
        ...
```

## 13.5 Playback Interface

```python
class PlaybackController:
    async def play(self, audio: "AudioCandidate") -> None:
        ...

    async def stop_immediately(self) -> None:
        ...

    def is_playing(self) -> bool:
        ...
```

---

# 14. 推奨ディレクトリ構成

```text
predictive-speaking/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ configs/
│  ├─ default.yaml
│  ├─ azure_openai.yaml
│  ├─ asr_qwen3.yaml
│  └─ tts_qwen3.yaml
├─ src/
│  └─ predictive_speaking/
│     ├─ app.py
│     ├─ config.py
│     ├─ audio/
│     │  ├─ capture.py
│     │  ├─ playback.py
│     │  ├─ ring_buffer.py
│     │  └─ vad.py
│     ├─ asr/
│     │  ├─ base.py
│     │  ├─ qwen3_asr.py
│     │  └─ mock_asr.py
│     ├─ prediction/
│     │  ├─ base.py
│     │  ├─ azure_openai.py
│     │  ├─ prompts.py
│     │  ├─ validator.py
│     │  └─ scheduler.py
│     ├─ tts/
│     │  ├─ base.py
│     │  ├─ qwen3_tts.py
│     │  ├─ cosyvoice.py
│     │  └─ mock_tts.py
│     ├─ voice_conversion/
│     │  ├─ base.py
│     │  └─ beatrice.py
│     ├─ hesitation/
│     │  ├─ detector.py
│     │  └─ rules.py
│     ├─ orchestration/
│     │  ├─ state_machine.py
│     │  └─ session.py
│     ├─ logging/
│     │  ├─ events.py
│     │  └─ metrics.py
│     └─ ui/
│        └─ debug_ui.py
├─ scripts/
│  ├─ run_prototype.py
│  ├─ benchmark_asr.py
│  ├─ benchmark_llm.py
│  ├─ benchmark_tts.py
│  └─ replay_session.py
├─ tests/
│  ├─ test_stable_prefix.py
│  ├─ test_candidate_validator.py
│  ├─ test_state_machine.py
│  └─ fixtures/
└─ logs/
```

---

# 15. 設定ファイル案

```yaml
audio:
  input_sample_rate: 16000
  output_sample_rate: 24000
  frame_ms: 20
  input_device: null
  output_device: null

asr:
  backend: qwen3
  model: Qwen/Qwen3-ASR-0.6B
  device: cuda
  stable_history_size: 3
  min_prefix_growth_chars: 4
  max_update_interval_ms: 400

llm:
  backend: azure_openai
  deployment_name: ${AZURE_OPENAI_DEPLOYMENT}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  api_version: ${AZURE_OPENAI_API_VERSION}
  temperature: 0.1
  max_output_tokens: 10
  request_timeout_ms: 1500
  prediction_interval_ms: 350

tts:
  backend: qwen3
  model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  device: cuda
  speaker: null

voice_conversion:
  backend: beatrice
  enabled: true

hesitation:
  silence_trigger_ms: 450
  confirmation_delay_ms: 150
  min_speech_before_trigger_ms: 700
  cooldown_ms: 800

candidate:
  max_chars: 15
  ttl_ms: 1500
  reject_numbers: true
  reject_negation: true
  reject_question: true
```

---

# 16. 環境変数

`.env.example`

```bash
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT=

INPUT_AUDIO_DEVICE=
OUTPUT_AUDIO_DEVICE=

HF_TOKEN=
CUDA_VISIBLE_DEVICES=
```

秘密情報はGitへコミットしない。

---

# 17. ログ仕様

すべてのイベントにmonotonic timestampを付ける。

## 17.1 イベント例

```json
{"t_ms": 0, "event": "session_start"}
{"t_ms": 412, "event": "speech_start"}
{"t_ms": 865, "event": "asr_partial", "text": "このシステムでは"}
{"t_ms": 1120, "event": "stable_prefix", "text": "このシステムでは"}
{"t_ms": 1140, "event": "llm_request_start", "request_id": 3}
{"t_ms": 1278, "event": "llm_first_token", "request_id": 3}
{"t_ms": 1302, "event": "llm_complete", "request_id": 3, "text": "ユーザーの発話を"}
{"t_ms": 1310, "event": "tts_start", "candidate_id": "cand-3"}
{"t_ms": 1415, "event": "tts_first_audio", "candidate_id": "cand-3"}
{"t_ms": 1450, "event": "candidate_ready", "candidate_id": "cand-3"}
{"t_ms": 1790, "event": "silence_start"}
{"t_ms": 2240, "event": "hesitation_detected"}
{"t_ms": 2255, "event": "assist_playback_start", "candidate_id": "cand-3"}
{"t_ms": 2480, "event": "user_resumed"}
{"t_ms": 2505, "event": "assist_playback_stop"}
```

## 17.2 保存情報

- 生音声または任意保存
- ASR partial/final
- stable prefix
- LLM入力
- LLM出力
- 候補採否理由
- TTS生成時間
- 再生開始時間
- ユーザー再開時間
- APIエラー
- GPUメモリ
- CPU/GPU使用率

個人音声を保存する場合は、保存可否を設定で切り替えられるようにする。

---

# 18. 評価指標

## 18.1 レイテンシー

- `ASR_TTPartial`
- `LLM_TTFT`
- `LLM_Total`
- `TTS_TTFA`
- `TTS_Total`
- `CandidateReadyBeforeHesitation`
- `HesitationToPlayback`
- `UserResumeToStop`

p50、p90、p95、最大値を算出する。

## 18.2 品質

- 続き候補受容率
- 発話すべきでない場面での誤発火率
- 先頭1〜3モーラ一致率
- 候補が文脈に自然だった割合
- 質問・相槌を誤生成した割合
- 数字・固有名詞の誤生成率
- ユーザーによる即時中断率
- `<SILENCE>`率
- 補助が間に合った割合

## 18.3 主観評価

各候補についてユーザーに以下を記録できるとよい。

```text
0: 全く違う
1: 少し近い
2: 意図として近い
3: そのまま言おうとしていた
```

---

# 19. MVP実装順序

## Phase 1: オフライン疑似実行

- 録音済み音声を入力
- ASRは既存文字起こしまたはMock
- Azure OpenAIで継続候補を生成
- 候補検証
- TTS生成
- 各遅延を記録

目的：

- プロンプト品質確認
- Azureモデル比較
- TTS比較

## Phase 2: リアルタイムASR

- マイク入力
- ストリーミングASR
- stable prefix生成
- 発話中の先読み
- 旧リクエストキャンセル

## Phase 3: 候補音声キャッシュ

- LLM候補をTTSへ送る
- PCMをメモリへ保存
- 候補TTL
- 新候補による置換

## Phase 4: 言い淀みトリガー

- VAD
- 無音長
- 発話未完結判定
- 手動トリガーも用意

初期はキーボード操作による手動トリガーを残す。

## Phase 5: 再生・割り込み

- 詰まり時に候補再生
- ユーザー再開時に即停止
- cooldown

## Phase 6: Beatrice統合

- TTS出力をBeatriceへ入力
- 総遅延計測
- 声質確認

## Phase 7: 評価UI

表示項目：

- ASR partial
- stable prefix
- 現在候補
- 候補生成時刻
- TTS準備状態
- 言い淀み状態
- システム状態
- 各区間レイテンシー

---

# 20. MVPの受け入れ条件

以下を満たしたら最初のMVPとする。

1. マイク音声をストリーミング認識できる
2. 発話途中のstable prefixからAzure OpenAIが短い続きを生成できる
3. 古いLLM候補を破棄できる
4. 候補音声を詰まり前に生成できる
5. 手動トリガーで即時再生できる
6. ユーザーの発話再開で停止できる
7. 各処理のタイムスタンプをログ保存できる
8. 誤った長文応答や質問文を候補検証で拒否できる
9. Azure GPT-4.1 mini相当とGPT-4o mini相当を切り替えられる
10. TTSを少なくとも2方式で比較可能

---

# 21. 初期実験セット

最低100件程度の日本語発話断片を作る。

## 21.1 正例

```text
今回の研究では、ユーザーの発話を
→ リアルタイムに解析して

このシステムの最大の特徴は
→ 次の言葉を予測できる点です

音声認識の結果を使って
→ 発話の続きを生成します
```

## 21.2 沈黙すべき例

```text
本日はありがとうございました
→ <SILENCE>

この件についてどう思いますか
→ <SILENCE>

条件は三つあります
→ <SILENCE>
```

## 21.3 曖昧例

```text
私は昨日
→ <SILENCE>

今回使ったモデルは
→ <SILENCE>

実験参加者は
→ <SILENCE>
```

## 21.4 危険例

```text
費用は
→ <SILENCE>

薬の名前は
→ <SILENCE>

契約期間は
→ <SILENCE>
```

---

# 22. 比較実験

## 22.1 Azureモデル

同一入力セットで比較する。

- GPT-4.1 mini相当
- GPT-4o mini相当
- 利用可能なら他の低遅延miniモデル

測定：

- TTFT p50/p95
- 継続候補受容率
- `<SILENCE>`精度
- 質問や返答の誤生成率

## 22.2 TTS

- Qwen3-TTS 0.6B
- CosyVoice3 0.5B
- Style-Bert-VITS2
- Azure TTS

測定：

- Time To First Audio
- 短文自然性
- 日本語アクセント
- Beatrice通過後の自然性
- GPU負荷

## 22.3 ASR

- Qwen3-ASR 0.6B
- Qwen3-ASR 1.7B
- 既存ASR
- Azure Speech

測定：

- partial更新速度
- stable prefix確定速度
- 途中認識の巻き戻り量
- 日本語専門用語の精度
- GPU負荷

---

# 23. 実装上の注意

## 23.1 直列awaitを避ける

以下のようにしない。

```python
text = await asr()
prediction = await llm(text)
audio = await tts(prediction)
await play(audio)
```

ASR、LLM、TTS、VAD、再生を独立タスクとして動かし、イベント駆動で連携する。

## 23.2 Queueは最新値優先

途中結果は古いものに価値がない。

通常のFIFOではなく、

- queue size = 1
- 新値で旧値を上書き
- request_idで鮮度を管理

する。

## 23.3 音声スレッドをブロックしない

モデル推論やAPI呼び出しを音声コールバック内で実行しない。

音声コールバックはリングバッファへ書き込むだけにする。

## 23.4 時刻計測

wall clockではなく、Pythonなら`time.perf_counter_ns()`等のmonotonic clockを用いる。

## 23.5 タイムアウト

候補は遅れて届いても価値がない。

例：

```text
LLM timeout: 1000〜1500ms
TTS timeout: 1000ms
candidate TTL: 1500ms
```

タイムアウト時は沈黙する。

## 23.6 エラー時のフォールバック

- ASR停止 → 補助機能停止
- Azure失敗 → 発話しない
- TTS失敗 → 発話しない
- Beatrice失敗 → 設定により元TTS音声または無音
- GPU OOM → モデル停止とログ

適当なローカル候補を発話するフォールバックは初期段階では採用しない。

---

# 24. 未確定事項

Coding Agentは以下を実装前または初期調査で確認すること。

1. Qwen3-ASRの現行ストリーミングAPI
2. Qwen3-ASRが単一ストリーム低遅延用途で安定するか
3. Qwen3-TTSのtext-in streaming API
4. CosyVoice3との比較
5. Beatriceの最適なプログラム連携方法
6. WindowsとLinuxのどちらを主環境にするか
7. GPU割り当て
8. Azure OpenAIの利用API
9. AzureのDeployment Name
10. 現在利用可能なモデル
11. 音声入出力デバイス
12. 既存コードから流用可能な部分
13. ユーザー固有の言い淀み検出方法
14. シーン情報・発表原稿をどのように入力するか

---

# 25. 推奨GPU配置

ユーザー環境には以下がある。

- RTX A6000 48GB
- RTX 4070 Super
- 別環境としてDGX Spark相当機

推奨案：

```text
RTX 4070 Super:
- ASR
- TTS
- Beatrice

RTX A6000:
- 将来のローカルLLM比較
- 重いモデルの検証
```

Azure OpenAIを使用する場合、A6000をLLMへ割り当てる必要はない。

一枚運用の場合は、ASR・TTS・BeatriceのGPU競合を計測する。

---

# 26. 将来拡張

- ユーザーの発話履歴から個人語彙を抽出
- 発表スライドや原稿を文脈として使用
- Retrievalによる候補空間制限
- 話者ごとの言い回し辞書
- 候補の複数生成とconsensus
- 文脈別プロンプト
- 短い補助と長い補助の切り替え
- UI上で候補選択
- 視線、ジェスチャー、キー入力による承認
- LLM-jp-MoshiやKAME系との比較
- 音声のみモデルへの置換
- 小規模Adapter/LoRA
- ユーザーの採用／拒否ログによる学習

---

# 27. Coding Agentへの実装指示

まず、フルシステムを一度に作らず、以下の順に実装すること。

1. **Azure OpenAI継続予測の単体CLI**
2. **候補検証器**
3. **録音音声を使ったTTS先行生成**
4. **イベントログと遅延計測**
5. **Mock ASRを使った状態機械**
6. **実ストリーミングASR**
7. **手動トリガーによる候補再生**
8. **VADベースの自動トリガー**
9. **ユーザー再開による割り込み停止**
10. **Beatrice統合**

各段階で単体テストを作り、モデル固有コードをインターフェースの内側へ閉じ込めること。

モデルを交換しても、オーケストレーション層を変更しない設計にすること。

---

# 28. 最初に作るCLIの仕様

コマンド例：

```bash
python scripts/benchmark_llm.py \
  --input examples/prefixes.jsonl \
  --deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --output logs/llm_benchmark.jsonl
```

入力：

```json
{"id":"001","prefix":"このシステムではユーザーが言葉に詰まったとき","expected":"次の言葉を予測して","should_silence":false}
{"id":"002","prefix":"本日はありがとうございました","expected":"","should_silence":true}
```

出力：

```json
{
  "id": "001",
  "candidate": "次の言葉を予測して",
  "ttft_ms": 142,
  "total_ms": 188,
  "accepted": true,
  "reject_reason": null
}
```

---

# 29. Definition of Done

プロトタイプ完成時、以下を満たすこと。

- 発話中に候補生成が走る
- 候補音声が詰まり前に準備される
- 詰まり検出時に即時再生される
- ユーザー再開時に即停止する
- 不確実な場合は沈黙する
- AzureモデルとTTSを設定で交換できる
- すべての遅延をログから再現できる
- 100件以上の発話断片で比較評価できる
- 失敗時に誤った補助音声を出さない
- READMEにセットアップと実行方法が記載される

---

# 30. 最終方針

今回のプロトタイプでは、低遅延化のために精度の低いローカルLLMへ全面移行しない。

採用する方針は以下である。

```text
高品質なAzure OpenAIを発話中から先読み実行し、
ASR・LLM・TTSの処理時間をユーザーの通常発話時間へ隠す。
詰まり発生時には、生成済み候補音声のみを解放する。
```

PredictiveSpeakingでは、単純な生成速度よりも以下を優先する。

1. 発話すべきでない場面で話さない
2. 同じ話者の続きを生成する
3. 短い候補だけを出す
4. ユーザーが再開したら止まる
5. 誤予測を本人の声で発話しない
6. 候補が間に合わない場合は沈黙する

以上を、実装・評価の最重要原則とする。
