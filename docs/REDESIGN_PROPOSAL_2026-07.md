# PredictiveSpeaking 再設計提案（2026-07-30 時点の技術調査に基づく）

本書は、旧実装（`../PredictiveSpeaking_old`）と引き継ぎ資料（`PredictiveSpeaking_handoff.md`）を踏まえ、
2026年7月時点で実際に使える技術で再構築する場合の推奨構成をまとめたものである。
技術情報は Web 調査（2026-07-30 実施）で個別に検証済み。未検証事項は末尾に明記する。

---

## 1. 結論サマリ

**handoff の「先読み・キャッシュ・ゲート」アーキテクチャはそのまま採用する。変えるのはモデル選定である。**

| コンポーネント | handoff案 | 本提案（第一候補） | 理由 |
|---|---|---|---|
| ASR | Qwen3-ASR-0.6B | **Azure Speech ストリーミング（MVP）→ NVIDIA Nemotron-3.5-ASR-Streaming-0.6B（ローカル化）** | Qwen3-ASRのストリーミングは約2秒チャンク・vLLM限定でpartial更新が遅すぎる |
| LLM | Azure OpenAI gpt-4.1-mini | **ローカル: Qwen3 Swallow 8B（CPT版）+ llama.cpp** （クラウド比較: gpt-5.4-mini / Gemini 3.5 Flash-Lite） | gpt-4.1-miniは非推奨化済み・Azure TTFT実測~1s。日本語ローカル小型モデルの品質が2026年時点で逆転した |
| TTS | Qwen3-TTS-12Hz-0.6B-CustomVoice + Beatrice | **Qwen3-TTS-12Hz-Base（クローン版）単体** — 旧リポジトリの自作ストリーミングサーバを改良して続用 | CustomVoiceはクローン不可。BaseはApache-2.0でゼロショットクローン可能、VCが不要になる |
| Voice Conversion | Beatrice | **廃止（メインパスから外す）** | Beatriceは公式にはVSTのみでプログラム連携手段がない。TTS直接クローンで代替可能 |
| 詰まり検出 | VAD + ルール | **TEN VAD + ルール + 京大 MaAI（VAP）** | MaAIは「話者が続けるか」を連続予測する日本語対応モデルで、まさにこの用途向け |
| オーケストレーション | Python asyncio 自作 | **同左（handoff の状態機械をそのまま実装）** | pipecat等のフレームワークは対話用途向けで、本件は自作が最も単純 |

さらに、ローカルLLM化によって handoff にない **二段構え予測**（3.2節）が可能になり、
候補キャッシュのヒット率問題を大きく緩和できる。

---

## 2. 前提の整理

### 2.1 旧実装の本質的な問題

旧実装（Node.js + ブラウザ）を読んだ結果、遅延の主因はモデル速度ではなく**アーキテクチャ**である。

- 予測が ASR の **final 結果が出てから** しか始まらない（`server.js handleTranscript`）
- 予測 → TTS → 音声転送 が完全直列
- 言い淀み検出・再生ゲートが存在せず、「詰まったときだけ話す」制御がない
- chat モデルに継続を頼むため「アシスタント応答」が混入し、検出→strictリトライで更に遅延
  （`azurePredictor.js isLikelyAssistantReply` + リトライ）

handoff はこの4点すべてに正しく対処している（先読み・並列化・ゲート・検証）。設計思想は維持すべき。

### 2.2 旧実装から再利用できる資産

| 資産 | 場所 | 再利用方法 |
|---|---|---|
| Qwen3-TTS ボイスクローン・ストリーミングサーバ（初回チャンク~208ms、クローンプロンプトキャッシュ、torch.compile済み） | `voice_clone_streaming_server.py` | **ほぼそのまま新TTSバックエンドに**。faster-qwen3-tts系の最適化（StaticCache + CUDA Graphs）を取り込む |
| 継続予測プロンプト・禁止パターン（アシスタント応答検出正規表現等） | `backend/azurePredictor.js` | Validator へ移植（ローカルbase modelでは発生頻度が激減するが保険として） |
| speakable 判定・日本語長さ調整 | `server.js sanitizePredictionForSpeech` ほか | Validator へ移植 |
| llama-server 運用ノウハウ（Qwen3 4B/14B/32B、structured prompt） | `localTransformerPredictor.js`, `.env.example` | ローカルLLM predictor の下敷きに |
| 日本語継続の評価スクリプト | `scripts/eval_continuation_ja.js` | Python へ移植し Phase 0 のベンチに |
| 予測IDによる陳腐化破棄 | `server.js currentPredictionId` | request_id 方式としてそのまま |

---

## 3. 提案アーキテクチャ

### 3.1 全体像

```text
[Mic 16kHz] ── リングバッファ（音声コールバックは書き込みのみ）
   ├─→ TEN VAD ──────────────→ 無音開始/発話開始イベント（数ms、CPU）
   ├─→ MaAI (VAP) ───────────→ 「話者継続確率」ストリーム（CPU、日本語対応）
   └─→ Streaming ASR（差し替え可能IF）
          MVP: Azure Speech / ローカル化: Nemotron-3.5-ASR-Streaming-0.6B
        ↓ partial
   [Stable Prefix Tracker] 直近2〜3partialのLCP、3〜5文字増加 or 400ms経過でflush
        ↓ stable prefix（+ 最新partialも保持）
   [Prediction Scheduler] request_id採番・旧リクエスト破棄
        ↓
   [ローカルLLM] Qwen3 Swallow 8B CPT / llama.cpp（A6000）
        n=2〜3サンプリング + logprobs
        ↓
   [Candidate Validator] 候補間一致度・logprob信頼度・数字/固有名詞/否定/完結文の拒否
        ↓ 承認候補
   [Qwen3-TTS Base クローン] 先行合成 → PCMキャッシュ（TTL 1.5s）
        ↓
   [Candidate Audio Cache]（本人声のPCMが既に完成している）

[Hesitation Detector]（VAD無音350–500ms × prefix未完結 × VAP継続確率低下 × フィラー語彙 × cooldown）
        ↓ trigger（+100–200ms確認猶予）
   [Playback Gate] → WASAPI排他モードで即時PCM再生
   [Barge-in] VADが発話再開を検知 → 100ms以内に再生停止 → COOLDOWN
```

handoff の状態機械（IDLE〜COOLDOWN）・候補鮮度管理・ログ仕様・評価指標は**そのまま採用**する。

### 3.2 二段構え予測（handoff からの最重要追加）

handoff 設計の弱点は「詰まった瞬間にキャッシュ済み候補が新鮮とは限らない」こと。
stable prefix が400ms前の状態で作った候補は、ユーザーが実際に詰まった位置と食い違いうる。
クラウドLLM（TTFT~1s）ではこのミス率が高いが、**ローカルLLM（合計~300ms）なら救済パスが作れる**。

1. **投機層（常時）**: stable prefix 更新のたびに候補生成 → TTS → キャッシュ（handoff通り）
2. **確定層（詰まり検出時）**: HESITATION_PENDING に入った瞬間、**最新partial全文**から
   再予測を発火。確認猶予（100–200ms）+ 再生立ち上がりの間に間に合えば、
   キャッシュ候補と照合して「最新partialと矛盾しない方」を再生。間に合わなければ
   キャッシュ候補（鮮度チェック通過時のみ）、それもなければ沈黙。

これは llama.cpp のプロンプトキャッシュ（会話履歴部分は毎回同一）で TTFT を数十msに抑えられる
ローカル運用だからこそ成立する。TTSも投機層で声・話速が同一の候補を作ってあるため、
確定層ではテキストが一致すればキャッシュ再生、僅差の場合のみ再合成（~200ms）となる。

### 3.3 「話さない」判定の強化（ローカルならではの材料）

- **n-bestの分散**: n=3 サンプリングで候補が割れたら `<SILENCE>`（handoff の「候補分散→沈黙」ルールが実装可能になる）
- **logprob信頼度**: 先頭トークンの対数尤度が閾値未満なら沈黙。クラウドAPIでは取りにくい情報
- 数字・固有名詞・否定・完結文の拒否ルールは handoff §9 の通り

---

## 4. 技術選定の根拠（2026-07-30 調査）

### 4.1 ASR — handoff 第一候補は不採用

- **Qwen3-ASR-0.6B/1.7B**: 2026-01 に Apache-2.0 でオープン化されたのは事実。ただし
  ストリーミングは公式 vLLM フレームワーク限定で**約2秒チャンク**駆動。partial 更新間隔
  300ms 以下という本件要件に合わない。sherpa-onnx 統合はオフライン専用。
  （https://github.com/QwenLM/Qwen3-ASR）
- **ReazonSpeech k2 v2 / kotoba-whisper**: 日本語品質は最高クラスだが、いずれも実体はオフライン
  モデル（チャンク再デコード運用）。真の増分ストリーミングではない。
- **採用（ローカル）: nvidia/nemotron-3.5-asr-streaming-0.6b**（2026-06公開）
  - cache-aware RNN-T による**真の増分ストリーミング**、チャンク 80/160/320/560/1120ms 可変
  - 日本語対応（40言語）、寛容ライセンス（OpenMDW-1.1）
  - 懸念: ja CER 11.5%（1.12sチャンク時）と品質は中位。短チャンク時の精度は未公表。
    NeMo/NIM は Linux 前提 → **WSL2 か Docker コンテナで動かし localhost WebSocket で接続**
  - https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- **採用（MVP）: Azure Speech ストリーミング** — 既存Azure契約で即使え、日本語品質が高く、
  2026年に「Post-Stream Refinement」（即時partial + 高精度final）が追加された。
  まず Azure Speech でオーケストレーション全体を完成させ、ASR IF 越しに Nemotron を差し替え評価する。
- 補欠（クラウド低遅延）: Soniox v5（sub-300ms実測を公表）、Deepgram Flux Multilingual ja
  （ターンテイキングイベント内蔵）。
- 品質が問題になる場合の発展形: **二層ASR**（Nemotronのpartialはタイミング・prefix用、
  ReazonSpeech k2 v2 のオフライン再デコードを会話履歴の清書用に併走）。

### 4.2 LLM — handoff の「ローカル不採用」判断を覆す

handoff がローカルLLMを外した理由は「軽量モデルの日本語継続品質が弱い」だったが、状況が変わった。

- **Azure 側の変化**: gpt-4.1-mini は**非推奨化済み**（2027-04-14 retirement、既存デプロイは継続可）。
  gpt-4o-mini も同様。後継の低遅延枠は gpt-5.4-mini/nano（2026-03、reasoning_effort: minimal 必須）。
  ただし Azure 経由 TTFT 実測は gpt-4.1-mini ~1.46s / gpt-5.4-mini ~1.04s（米国計測、日本からはRTT加算）。
  **~1秒のTTFTでは stable prefix（400ms間隔更新）に対して候補が常に1世代古くなる。**
- **ローカル側の変化**: **Qwen3 Swallow 8B**（東工大/AIST、2026-02、Apache-2.0）が
  同クラス日本語オープンモデルの最高水準。しかも **CPT（継続事前学習）チェックポイント**が
  公開されており、chatテンプレート無しの**素のテキスト継続**として使える。
  「同一話者の続きを予測する」タスクは本来 base モデルの言語モデリングそのものであり、
  旧実装を悩ませた「アシスタント応答混入」問題がカテゴリごと消える。
  （https://swallow-llm.github.io/qwen3-swallow.ja.html）
- A6000 48GB で 8B（AWQ/GGUF Q8）なら余裕。単発リクエストの実測公表値はないが、
  500トークン級プロンプト + 10トークン出力で **TTFT 200ms未満・合計150–400ms** が現実的な見積り。
  プロンプトキャッシュ（履歴部分固定）でさらに短縮可能。
- サービング: MVPは **llama.cpp（Windowsネイティブ、旧リポジトリで運用実績あり）**。
  スループットが要るなら vLLM（WSL2）へ。
- クラウドは**設定で差し替え可能な比較対象として残す**: gpt-5.4-mini/nano（minimal）、
  Gemini 3.5 Flash-Lite（2026-07、~350 tok/s、ただし日本リージョン提供状況は要確認）。
  Groq は Llama系を廃止済みで日本POPも無く優先度低。

### 4.3 TTS — Qwen3-TTS 続用、ただし Base（クローン版）

- **Qwen3-TTS ファミリーは 2026-01-22 に Apache-2.0 でオープン化**。ただし handoff 指定の
  **CustomVoice はプリセット話者のみでクローン不可**。本人声にするのは **Base**（0.6B/1.7B）で、
  約3秒の参照音声からゼロショットクローンできる。（https://github.com/QwenLM/Qwen3-TTS）
- 旧リポジトリの `voice_clone_streaming_server.py` は既に 1.7B-Base +ストリーミング+クローン
  プロンプトキャッシュで初回チャンク~208msを達成しており、**この構成は2026年7月時点でも
  ほぼ最適解**。community 最適化（faster-qwen3-tts: StaticCache + CUDA Graphs、RTX 4090で
  TTFA 152ms/0.6B）を取り込み、0.6B-Base との品質・速度比較を行う。
- 比較対象: Fun-CosyVoice3-0.5B（2025-12、Apache-2.0、~150msストリーミング公称）、
  Style-Bert-VITS2 JP-Extra（本人音声でファインチューンすれば話者再現の上限は最も高い。
  MOS 4.37 ≒ 人間。ただし AGPL/LGPL と学習の手間）、Aivis Cloud（~0.3s）。
- 先行生成キャッシュ方式のため TTFA は再生遅延там直結しないが、**候補の鮮度窓
  （TTL 1.5s）内に合成を終える**必要があるため速いほどキャッシュヒット率が上がる。

### 4.4 Voice Conversion — Beatrice をメインパスから外す

- Beatrice 2.0（rc）は ~50ms・CPU動作と性能は優秀だが、**公式提供は VST プラグインのみ**。
  CLI/Python/REST の公式APIは存在せず、推論ライブラリ（beatrice.lib）の独立利用条件も不明瞭。
  プログラム制御が前提の本システムには統合コストとライセンスリスクが大きい。
  （https://prj-beatrice.com/）
- Qwen3-TTS Base の直接クローンで「本人に近い声」を出せるため、**VC段そのものを削除**して
  パイプラインを1段短縮する。
- クローン品質が不足した場合の代替（優先順）:
  1. Style-Bert-VITS2 を本人音声でファインチューン（TTS自体を本人声にする）
  2. Beatrice を使うなら「リアルタイム変換」ではなく**キャッシュ済みPCMへの事前一括変換**
     として組み込む（先読みアーキテクチャならVC遅延も隠せる）。経路は beatrice-trainer
     （MIT）で本人モデルを学習し、w-okada VCClient の非公式REST等で変換
  3. seed-vc はリポジトリがアーカイブ済み（2025-11）のため非推奨

### 4.5 言い淀み検出 — 今回の調査での最大の収穫

- **MaAI（京都大学、`pip install maai`、MIT、Windows対応・CPUリアルタイム）**:
  Voice Activity Projection（VAP）モデルで「この話者がこの後も発話を続けるか」を
  フレーム単位で連続予測する。**ターン終了検出ではなく発話継続予測**であり、
  「詰まり＝続けたいのに止まっている」を判別したい本件に現状最も適合する。
  日本語学習済み（日本語対話コーパス）。相槌タイミング予測も持つ。
  （https://github.com/MaAI-Kyoto/MaAI）
- **TEN VAD**（Agora、2025-06 OSS化）: 306KB・RTF 0.015。Silero が数百ms遅れると指摘される
  **speech→無音の遷移検出が速い**ことを売りにしており、「無音開始を早く知る」本件の
  トリガー起点に向く。保険として Silero VAD v6 系と差し替え可能にする。
- フィラー（えっと/あの/その…）は専用モデルではなく **ASR partial の語彙スポッティング**で
  拾う（Whisper系・Azureともフィラーをある程度書き起こす。専用の日本語フィラー検出
  既製モデルは見つからず）。
- pipecat smart-turn v3 / LiveKit turn detector は日本語対応だが「ターン終了」検出器であり、
  本件の主判定には向かない（補助特徴としては将来検討可）。
- 判定式は handoff §11 のルール（無音350–500ms × prefix未完結 × 候補準備済み × cooldown）を
  土台に、**VAP継続確率を追加条件**として誤発火を削る。初期は手動トリガー併設（handoff通り）。

### 4.6 オーケストレーション

- 対話エージェント用フレームワーク（pipecat / LiveKit Agents / TEN）を検討したが、
  本件は「対話」ではなく単一話者の補助であり、handoff の状態機械を **Python asyncio で
  自作するのが最短**。フレームワークからは部品（VAD、割り込み処理の設計パターン）だけ借りる。
- handoff §13 のインターフェース定義（StreamingASR / ContinuationPredictor / StreamingTTS /
  PlaybackController）と §14 のディレクトリ構成はそのまま使う。VoiceConverter IF は
  「任意の後処理段」として残すが既定は無効。

---

## 5. プロセス・GPU配置

```text
Windows ネイティブ（メインプロセス, Python asyncio）
  - 音声入出力（sounddevice/WASAPI排他, フレーム20ms）
  - TEN VAD / MaAI（CPU）
  - Stable Prefix / Scheduler / Validator / 状態機械 / ログ
  - llama.cpp サーバ（A6000, Qwen3 Swallow 8B CPT）… localhost HTTP
  - Qwen3-TTS Base ストリーミングサーバ（4070 Super）… 旧サーバ改良, localhost HTTP

WSL2 / Docker（ローカル化フェーズで追加）
  - Nemotron-3.5-ASR-Streaming（NeMo/NIM, 4070 Super か A6000 の空き）… localhost WebSocket

クラウド（MVP と比較用）
  - Azure Speech（ASR MVP）
  - gpt-5.4-mini / Gemini 3.5 Flash-Lite（LLM比較用バックエンド）
```

- GPUメモリ概算: LLM 8B量子化 ~6–10GB（A6000に余裕）、TTS 0.6B/1.7B ~4–12GB、
  ASR 0.6B ~2–4GB。4070 Super（12GB）に TTS + ASR を同居させる場合は競合を実測する
  （handoff §25 の方針通り）。

---

## 6. レイテンシバジェット（目標: 詰まり検出→再生 p50 150ms / p95 350ms）

| 区間 | 見積り | 備考 |
|---|---|---|
| 無音開始 → VAD検出 | 20–60ms | TEN VAD、20msフレーム |
| 無音継続判定 | 350–500ms | 「詰まり」の定義そのもの（遅延ではなく仕様） |
| 確認猶予 | 100–200ms | この間に確定層の再予測を並走（3.2節） |
| キャッシュPCM解放 → 出音 | 10–30ms | WASAPI排他・プリオープン済みストリーム |
| **（キャッシュヒット時の体感遅延）** | **~130–290ms** | 無音判定完了を0点とした場合 |
| 投機層: prefix→候補音声完成 | 400–800ms | LLM 150–400ms + TTS TTFA~200ms+尾部。通常発話中に完了させる |
| バージイン: 発話再開→停止 | <100ms | VAD onset→バッファクリア |

キャッシュヒット率（=詰まり時に有効候補が存在する確率）が最重要KPI。
クラウドLLM(TTFT~1s)では投機層が1.5–2s級になりTTL内に収まりにくいのに対し、
ローカルなら1世代あたり~0.5–1sで回り、prefix更新2回に1回は新鮮な候補を持てる計算。

---

## 7. 実装フェーズ（handoff §27 を改訂）

1. **Phase 0 — 継続予測の単体ベンチ（最初にやる価値が最大）**
   Qwen3 Swallow 8B CPT（llama.cpp）vs gpt-5.4-mini vs（残っていれば）gpt-4.1-mini を、
   handoff §21 の評価セット（正例/沈黙/曖昧/危険 100件）で比較。
   計測: TTFT / 合計 / 受容率 / `<SILENCE>`精度 / アシスタント応答混入率。
   旧 `eval_continuation_ja.js` を Python 移植して流用。
   **ここでローカルLLM案の成否が決まる。** ダメなら handoff 通り Azure 構成に戻す。
2. Phase 1 — Validator + n-best分散/logprob沈黙判定（オフライン）
3. Phase 2 — TTSサーバ改良（faster-qwen3-tts最適化取り込み、0.6B/1.7B比較）+ 先行生成キャッシュ
4. Phase 3 — Azure Speech ストリーミング + Stable Prefix + Scheduler（リアルタイム先読み成立）
5. Phase 4 — TEN VAD + ルール詰まり検出（手動トリガー併設）→ MaAI を追加して誤発火比較
6. Phase 5 — 再生ゲート + バージイン + cooldown + 全ログ/計測
7. Phase 6 — ASRローカル化（Nemotron/WSL2）と品質比較、必要なら二層ASR
8. Phase 7 — （クローン品質不足の場合のみ）SBV2ファインチューン or Beatrice事前変換

MVP受け入れ条件・ログ仕様・評価指標は handoff §17–20 をそのまま使う。

---

## 8. リスクと未検証事項

- **Nemotron ja 品質**: CER 11.5%（1.12sチャンク）は中位。短チャンク時の精度未公表。
  → Phase 6 で実測。ダメなら Azure Speech 継続 or Soniox v5 / 二層ASR。
- **A6000 での単発TTFT**: 公表ベンチは同時50リクエスト時のみ（TTFT ~600ms）。
  単発では大幅に短いはずだが実測が必要（Phase 0 で判明する）。
- **Qwen3-TTS の A6000/4070S 実測TTFA**: 4090 の 152ms は参考値。旧サーバの208msが下地。
- **MaAI の「詰まり」適合度**: VAP はターンテイキング研究由来であり、word-finding pause を
  どこまで拾えるかは要実験（Phase 4 でルール式とのA/B）。
- **gpt-4.1-mini の退役**: 2027-04-14。Azure比較枠は gpt-5.4-mini/nano へ移行前提で書く。
- **日本からのクラウドTTFT**: 公表値は米国計測。Groq/Cerebras等は+100–200ms見込み。
- 本件と同一の「日本語・同一話者・リアルタイム発話継続」の先行製品/研究は調査範囲では
  見つからず（AAC研究はテキスト入力補助が中心）。ベンチマーク対象がない＝自前評価が全て。

---

## 9. handoff からの変更点一覧（差分だけ知りたい人向け）

1. ASR: Qwen3-ASR → Azure Speech（MVP）/ Nemotron-3.5-ASR-Streaming（ローカル）
2. LLM: Azure gpt-4.1-mini 主軸 → **ローカル Qwen3 Swallow 8B CPT 主軸**（Azureは比較枠）
3. LLM形式: chat + system prompt → **base モデルの素のテキスト継続**（アシスタント応答問題の根絶）
4. TTS: Qwen3-TTS CustomVoice → **Qwen3-TTS Base（ゼロショットクローン）**、旧自作サーバ続用
5. VC: Beatrice → **削除**（不足時のみ SBV2 FT or キャッシュへの事前一括VC）
6. 詰まり検出: VAD+ルール → **TEN VAD + ルール + MaAI(VAP)**
7. 追加: **二段構え予測**（投機キャッシュ + 詰まり瞬間の最新partial再予測）
8. 追加: n-best分散・logprob による沈黙判定
