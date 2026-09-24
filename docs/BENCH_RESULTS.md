# ベンチ結果ログ

実験の時系列記録(新しい順)。本文中のパスは当時のもの:
`tts_server/` → `server/`(chatterbox / piper_train / llm)、`--local` → 既定化、
Qwen3-TTS・Nemotron・`run_live.py`・`register_voice.py` は2026-09-24の整理で削除済み。

## 2026-09-03 — Piper本人声fine-tune (chatterboxクローン合成データ → VITS、ローカルCPU 80ms)

ユーザー指示: 実録音の代わりにchatterboxクローン声で学習データを合成し、neshime/muramotoの
2声をPiper(piper1-gpl)でfine-tune。

| 段階 | 結果 |
|---|---|
| データ合成 | CMU ARCTIC 1132文 → neshime 1126本 / muramoto 1105本(長さ異常を自動除外)、計69.7分、1.6s/文 |
| 学習 | lessac medium(epoch2164)から+250epoch、batch32(VRAM~11GB)、2.2it/s・16s/epoch ≒ 67分/声 |
| 書き出し | ONNX 60.6MB(+config json) |
| **ローカル合成(このPC CPU)** | **neshime: 73〜114ms / muramoto: 82〜98ms /文、RTF 0.02〜0.04**(市販lessac: 62〜91ms) |

TTS速度比較(同一3文、このPC): Piper 80ms ≪ Kokoro 330ms ≪ chatterbox-vllm(遠隔) ~1.3s。
発話ペースは学習元(chatterbox neshime/muramoto: 同文2.8〜3.7s)を忠実に再現
(Piper 2.4〜3.4s)。速くしたい場合はlength_scaleで調整可。
品質(本人声再現度)は耳判定待ち: logs/piper_smoke/(Piper) vs logs/ab_cbx/(学習元Chatterbox)。
E2E(typed): タップ+自動発話(無音841msで発火)合格。両声とも run_all.sh の自動書き出しで完成。

踏んだ罠(すべてスクリプト化済み, tts_server/piper/): scikit-build不足 / torch2.6+ weights_only
(PosixPath) / LightningCLIの旧hparams再解析(sample_bytes) / vLLM EngineCore孤児によるOOM /
export_onnxのonnxscript不足+dynamoエクスポータ非対応(dynamo=False強制で解決)。

## 2026-08-31 — chatterbox-vllm導入 (EN TTS約30-40%高速化、:18084)

randombk/chatterbox-vllm(t3をvLLM化)を導入し、torch版とA/B(同一テキスト・同一外部回線):

| | 合成時間(3文) | RTF |
|---|---|---|
| torch版 (18083) | 1244〜1916ms | 0.64〜0.82 |
| **vLLM版 (18084)** | **974〜1350ms** | **0.49〜0.55** |

- クローン声(me_en)も動作、初回のみ+0.4s(条件計算)。声はtorch版とvoices_en共有
- 上流の売り文句「5-10倍」は長文バッチの数字。短文単発はs3gen(波形生成、torchのまま)が
  律速のため~1.3-1.4倍
- **構築の罠2つ**: ①PyPI版は古くvllm 0.10非互換 → リポジトリclone+uv sync必須
  ②サーバでchatterbox_vllmを関数内import すると、vLLMのspawn子プロセス(__main__再import)に
  EnTokenizer登録が伝わらずEngineCore起動失敗 → **importはトップレベル必須**
- 品質A/B用サンプル: logs/ab_torch/ vs logs/ab_vllm/(耳での判定はユーザー待ち。
  上流によるとspeech positional embeddings未適用だが影響軽微とのこと)
- アプリ: `--tts-backend chatterbox-vllm` で選択可
- VRAM整理: torch版chatterbox(18083)停止・ja LLM(18080)復旧。en LLM(18082)は停止のまま
  (ENはローカルLLM運用のため不要。必要なら8B-Baseを再起動)。44.6/49.1GB

## 2026-08-29 — EN品質の定量化: 「怪しさ」の正体は文脈不足(モデル容量ではない)

judge英語化(`judge_bench.py --lang en`、Swallow-8B-SFT@18081をEN促進文で使用)を実装し、
「英語予測が怪しい」を2つの実験で分解した。

### 実験1: モデル容量ラダー(en_utterances=履歴1文, raw n=3 np24 ※local-4bのみn=1)

| モデル | 逸脱率(coh=0) | coherence平均 | 意図一致(int>=1) | hit@1 |
|---|---|---|---|---|
| **ローカル4B-instruct Q4 (n=1)** | **34.1%** | 1.13 | 44.4% | 47.6% |
| 8B-Base Q8 | 38.9% | 1.04 | 42.0% | 49.6% |
| 4B-Base Q4 | 41.7% | 1.01 | 42.4% | 46.2% |
| 1.7B-Base Q8 | 44.3% | 0.94 | 39.7% | 44.3% |

**容量を上げても逸脱率はほぼ動かない**(むしろローカル4B-instructが最良)。
8B化はhit@1+3ptのみで、体感品質(逸脱)のレバーではない。

### 実験2: 文脈アブレーション(en_long_sessions.jsonl新設: 履歴400-600字×6本, 8B, 同一テキスト)

| | 逸脱率 | coherence平均 | 意図一致 | hit@1 |
|---|---|---|---|---|
| 履歴なし | 52.8% | 0.69 | 27.8% | 25.0% |
| **履歴あり(400-600字)** | **25.0%** | 1.11 | 38.9% | 44.4% |

**文脈で逸脱率が半減** — 日本語(31.8→7.1%)と同じ法則がENでも成立。
ジャッジは未校正のため相対比較用(ja数値との絶対比較は不可)。

### 結論と含意

- ENの「怪しさ」の主因は**予測に渡る文脈の薄さ**。短いテストセッション+話題メモ未使用の
  状態は最悪条件だった(アプリは履歴1200字まで自動蓄積するので、長く話すほど改善する)
- 対策の優先度: ①話題メモを使う(即効・実装済み) ②セッションを続けて履歴を貯める
  ③ASR誤認識の低減(Azureフレーズリスト等) — モデル交換は優先度低
- ローカル8B-Base Q4は選択肢として配置(--local-llm-model指定)だが既定は4Bのまま。
  ローカル実測(n=1, n_probs=0): **8B Q4 = 868ms / hit@1 49.2%**(Q8比の量子化劣化なし)
  vs 4B = 552ms / 47.6%。+316msでhit@1+1.6pt、逸脱率は改善しないため既定変更に値せず
- 運用メモ: judge(18081)再開に伴いChatterbox(18083)を停止(EN TTSはローカルKokoro移行
  済みのため。再開は`tts_server/setup_chatterbox.sh`)

## 2026-08-24 — ASRもローカル化: Vosk EN採用で`--local`が完全オフラインに

Whisper系はバッチエンジンでストリーミングpartial不可のため対象外(ja検討時と同じ結論)。
真のストリーミングASRとしてVosk en-us-0.22-lgraph(128MB)を導入しベイクオフ:

| (SAPI Zira 12.6s音源) | 初回partial | 間隔p50 | 間隔p90 | 認識 |
|---|---|---|---|---|
| Azure en-US | 1254ms | 491ms | 609ms | 完璧 |
| **Vosk en lgraph (ローカル)** | 3720ms | **345ms** | 706ms | **Azureと完全一致** |

- 定常partialはAzureより速い(345ms vs 491ms)。初回発話のみKaldiウォームアップで遅い
- VoskASRをja/en両対応化(enはスペース保持、既定モデルパスmodels/)+start()でモデル
  ロード完了待ち(ロード~5s中にマイク音声がキューあふれで欠落するのを防止)
- `--local`のASR既定をvoskに(--asr azureで上書き可)。実声品質は未検証(SAPI音源では同等)

## 2026-08-24 — 英語オールローカル実装完了 (`--local`)

`run_app.py --lang en --local` を実装しE2E合格。nessie不要・回線非依存。

- LLM: ローカルllama-server自動起動(Vulkan, Qwen3-4B Q4, n=1, **n_probs=0**)
  → E2E実測 wall p50=**552ms**(A/B予測通り。LAN経由A6000の615msと同等)
- TTS: Kokoro-82M **fp32+CPU固定**でRTF 0.23-0.39(562-881ms/文)。
  経緯: int8はCPUでRTF~1.0(逆に遅い)、DirectMLはint8/fp32とも実行時クラッシュ
  (onnxruntime-directml 1.24, UnicodeDecodeErrorに化けるop非対応エラー)→CPUピン留めが正解
- E2E: タップ(12サイクル0エラー)+自動発話(無音909msで発火 — 遠隔時3s超から短縮)合格
- 既知のアーティファクト: typedモードは語中でpartialを切るため「ethings about」型の
  語中開始候補が出る。実Azure ENは単語境界partialのため実使用では稀
- 依存変更: kokoro-onnx追加、onnxruntimeをonnxruntime-directmlに置換
  (pyproject override-dependencies)。モデルはmodels/kokoro/(int8はフォールバック残置)

## 2026-08-24 — 遠隔(Tailscale)自動発話の不発分析 + 英語オールローカル検証

### 遠隔で自動発話が出ない原因(実セッションログ)

外出先3セッションの分析: 回線劣化時にLLM往復が615ms→1136ms(最大1.9s)、TTS合成+転送が
750-830ms→1941ms(最大3.0s)となり、自動発話のブロック理由は`candidate_stale`が支配的
(最悪セッションで5/5件、自動発火0回)。ASR(Azure)はnessieを経由しないため無傷。
→ 対策: 鮮度判定に接頭辞許容(ja:2字/en:3字)を実装(webapp `_candidate_is_fresh`)。
冒頭重複は既存の`_spoken_overlap_remainder`が再生前に除去するため逸脱リスク増は小。

### 英語オールローカル(このWindows機/Radeon 8060S iGPU/Vulkan)の実測

**LLM品質(EN, raw n=3 np24, 22fixtures)**:

| モデル | hit@1 | hit@2 | 備考 |
|---|---|---|---|
| Qwen3-8B-Base Q8 (A6000) | 49.6% | 42.0% | 現行EN本番 |
| Qwen3-4B-instruct Q4 (ローカル実測) | 46.6% (n=1: 47.6%) | 39.7% | **手元に既存** |
| Qwen3-4B-Base Q4 | 46.2% | 40.2% | サーバで品質測定 |
| Qwen3-1.7B-Base Q8 | 44.3% | 38.9% | 8Bとの差5.3pt |

**重要発見: `n_probs=1`(信頼度logprobs)がこの機体では24トークンあたり~1秒のコスト**
(deocde 17→64 tok/s)。A6000(126コア)では無視できたためこれまで不可視だった。
iGPU素のデコード: 4B Q4=66.8 tok/s / 1.9GB級=79.7 tok/s (llama-bench)。

| ローカル4B Q4の1サイクル(24tok, n=1) | 実測 |
|---|---|
| n_probs=1 + DRY (現行設定) | 1521ms |
| **n_probs=0 + DRY** | **492ms** |
| 素(greedy のみ) | 366ms |

→ **n_probs=0にすればローカル4BでLAN経由A6000と同等速度(~500ms)+hit@1 47%**が成立。
失うのは信頼度表示(low_conf)のみ。1.7Bなら~350-400ms圏(未実測、帯域比から推定)。

### 英語オールローカル構成の結論(実装候補)

- LLM: ローカルllama-server(Vulkan) + 4B(既存gguf) n=1 n_probs=0 → ~500ms
- TTS: Kokoro-82M ONNX(CPU, RTF~0.3, 24kHz) — 声クローン不可だが現EN声はZiraプレースホルダ
  なので実質的損失なし。クローンが要る時だけ遠隔Chatterbox
- ASR: Azure継続(クラウド直・ローカル化と無関係)。完全オフライン化はsherpa-onnx(未検証)
- 期待効果: nessie依存が消え回線非依存。自動発話チェーン~1.3s(遠隔劣化時の3〜5sから大幅短縮)

## 2026-08-24 — 英語版アプリ完成 (Chatterbox TTS + langプロファイル + E2E)

`--lang en` 一発で英語構成(Azure en-US / Qwen3-8B-Base:18082 / Chatterbox:18083 /
`text/en.py`)に切り替わるプロファイルを実装し、E2E合格。

### Chatterbox TTS実測 (A6000, Windowsクライアントから)

| 声 | 合成時間 | 音声長 | RTF |
|---|---|---|---|
| default(内蔵) | 1365〜1584ms | 2.0〜2.8s | 0.57〜0.68 |
| me_en(クローン, 初回) | 3186ms | 2.1s | 1.50 |
| me_en(2回目以降) | 1382〜1585ms | 2.1〜2.3s | 0.67〜0.68 |

- 24kHz mono。Irodori(RTF≈0.25)の約2.5倍遅いが、先行合成キャッシュ方式なので
  タップ時点では合成済み — 体感への影響は「候補追従の鮮度」のみ
- 構築の罠: **perth(音声透かし)がpkg_resources依存** → setuptools 81以降で
  `PerthImplicitWatermarker=None`となりモデル初期化がTypeError。`setuptools<81`固定で解決
  (`setup_chatterbox.sh`に反映済み)

### EN E2E (typed 400ms刻み, en_utterances 6件)

- part2(タップ発話): 合格 — freshness p50=564ms / wall p50=484ms / 67サイクル0エラー
- part3(自動発話): 合格 — 無音3.1秒で「be ready by the end of the day」を自動発話→停止
- 予測品質は事前ベンチ通り(「to talk about our research」型の単語継続が素直に当たる)

### 実装メモ

- EN履歴結合バグをテストが検出: 句読点rstrip後に`en.is_incomplete`で完結判定していたため
  ピリオドが永遠に復元されなかった → 元発話の終止符で判定(jaは語尾形態判定なので無関係)
- `strip_leading_fillers`が「um, I think」のカンマ形を取り逃していた(Azure ENはカンマ挿入)→修正
- `validate()`のmax_chars既定をlang別解決に(en:60/ja:22 — 既定22のままだと英語が22字で切れる)
- テスト84件(EN追加13件)全合格

## 2026-08-18 — 英語版の実測 (LLM+ASR)

英語対応をベンチ基盤に追加(`text/en.py`、`validate(lang)`、`fixtures/en_utterances.jsonl`22件、
単語境界での予測位置、`--lang en`)。

### LLM英語トーナメント (raw n=3 np24、hit@kのみ、22fixtures×単語境界)

| モデル | hit@1 | hit@2 | wall p50 |
|---|---|---|---|
| **Qwen3-8B-Base Q8** | **49.6%** | **42.0%** | 576ms |
| Qwen3-14B-Base Q8 | 46.6% | 40.5% | 831ms |
| GPT-OSS-Swallow-20B(ja調整) | 45.7% | 40.3% | 588ms |
| gpt-oss-20b(素, MXFP4) | 42.1% | 34.9% | 527ms |

- **英語は根本的に予測が易しい**: 8B素モデルで日本語王者(20B)と同水準のhit@1、hit@2は42% vs 33%。
- **gpt-oss素版は継続タスクで意外に弱い**(instruct/推論特化でbase版が存在しないため)。
  日本語でSwallow版が勝ったのは「日本語SFTの付加」であって「gpt-ossの継続力」ではなかった。
- 英語の勝者: **Qwen3-8B-Base**(小さく速く最強 — 日本語より軽い構成で成立)。

### ASR英語ベイクオフ (SAPI Zira合成音声、同一入力)

| | partial間隔p50 | 認識品質 |
|---|---|---|
| Azure en-US | 499ms(この音源) | 完璧 |
| Nemotron-3.5 en-US | **235ms** | 可読だが誤り散見("test reporting for speed recogni") |

日本語で壊滅だったNemotronが英語では実用圏。品質はAzure優位のまま。
更新間隔重視ならローカル(Nemotron)が英語では初めて選択肢になる(実声での追試価値あり)。

### 英語版の残タスク

- **TTS載せ替えが必須**(Irodori-TTSはja特化)。候補: Chatterbox(MIT/クローン/低遅延),
  F5-TTS, クラウドならElevenLabs Flash(即席クローン・~135ms)。未実測。
- アプリのlangプロファイル(Azure言語・fillers/縫合のen切替・UI)、judgeの英語化。

## 2026-08-07 — ASRベイクオフ (Azure vs Nemotron-3.5-Streaming) → Azure継続

「ASRをもっと速く」の検証。faster-whisperはバッチエンジンのため対象外
(疑似ストリーミング化では実効0.5〜1s+書き換え揺れでAzureの295msより遅い)。

- 実装: NeMo環境(`asr_server/`)+WSブリッジ(`nemotron_ws.py`, port 18090)+クライアント
  (`asr/nemotron.py`)。`--asr nemotron`で切替可。ベイクオフは`scripts/bakeoff_asr.py`
- 落とし穴2つ: ①`EncDecRNNTBPEModelWithPrompt`は`set_inference_prompt("ja-JP")`必須
  (無指定だと無出力) ②入力音量が低いと無出力(me.webm原音はRMS 966で沈黙、×4増幅で出力)

| | partial間隔p50 | 実声(me.webm)の認識 |
|---|---|---|
| Azure Speech | 295ms(実セッション)/ 783ms(コールドスタート込みベイクオフ) | 「こんにちは今からテストの実行をします」(正確) |
| Nemotron 160ms | 243ms(SAPI wav) | **「その飛行き」(壊滅)** |

**結論: 不採用。** 速度ゲインは~50msに対し、実声での認識品質差が決定的
(誤認識1つで予測が脱線することは実セッションで実証済み)。チャンクを大きくすれば
精度は上がるがレイテンシ優位が消える。ハーネスは残置 — 将来の日本語ストリーミング
ASR(Soniox試行等)は`bakeoff_asr.py`で数分で判定可能。
現在の鮮度ラグ内訳はASR~300ms+LLM~615msでASRは主項ではない。

## 2026-08-06 — モデルトーナメント (「Swallowは最適か」の決着)

同一条件(raw n=3 np24 DRY、同一fixture180予測、同一ジャッジ)で7構成を比較。

| モデル | hit@1 | hit@2 | 逸脱率 | coherence | 意図一致 | wall p50 |
|---|---|---|---|---|---|---|
| **Swallow-8B-CPT Q8(現行)** | 38.1% | 24.3% | **26.7%** | **1.33** | **73.3%** | ~500ms |
| Qwen3-8B-Base Q8 | 35.8% | 19.9% | 32.4% | 1.21 | 66.5% | 570ms |
| Qwen3-14B-Base Q8 | 40.8% | 23.5% | 36.9% | 1.18 | 63.1% | 824ms |
| Swallow-30B-A3B-SFT Q4 | 41.1% | 24.6% | 30.3% | 1.23 | 69.1% | 438ms |
| Swallow-30B-A3B-CPT Q4 | 41.0% | **28.1%** | 31.5% | 1.26 | 68.5% | 1177ms |
| llm-jp-3-13B-instruct3 Q8 | 33.7% | 21.9% | 26.6% | 1.30 | 72.2% | 728ms |
| sarashina2.2-3B base Q8 | 23.6% | 12.7% | 27.9% | 1.30 | 71.5% | **224ms** |

### 追記2: GPT-OSS-Swallow-20B-SFT Q8 (ユーザー指摘で追加測定) — 全項目新記録・本番採用

| モデル | hit@1 | hit@2 | 逸脱率 | coherence | 意図一致 | wall p50 |
|---|---|---|---|---|---|---|
| **GPT-OSS-Swallow-20B-SFT Q8** | **49.7%** | **33.0%** | **24.0%** | **1.38** | **76.0%** | **615ms** |

- MoE(アクティブ~3.6B)のため**速度は8B並み**(615ms)で品質は32Bを大きく超える。両軸同時制覇。
- 「CPT版がないSFTは不利」という事前予想は**外れた**: gpt-oss(2025年世代)の基礎力+SwallowのSFTで、
  raw継続でもSFT税が出ない。CPT法則はQwen系内でのみ成立していた模様。
- 2026-08-06から**本番モデルをこれに変更**(8B CPTはフォールバックとして保持)。
- 未測定: GPT-OSS-Swallow-120B(MXFP4 ~63GBでA6000単体に載らず)。RL-v0.1版も未測定。

### 追記: Swallow-32B-CPT Q6_K (自前変換、ユーザー指摘で追加測定)

| モデル | hit@1 | hit@2 | 逸脱率 | coherence | 意図一致 | wall p50 |
|---|---|---|---|---|---|---|
| **Swallow-32B-CPT Q6_K** | **43.4%** | 27.4% | 26.9% | **1.34** | 72.6% | 1338ms |

**品質は全項目で首位級**(hit@1トップ、自然さも8B CPTと同率首位) — 「容量×日本語CPT×
良い量子化」の両取り仮説が的中。課題は速度のみ(8Bの~2.7倍)。
投機的デコード(8B Q4ドラフト、--spec-draft-n-max 12)を試したが**効果なし**(1341ms、
n_probs=0でも変化なし) — 現行llama-serverの並列スロット構成では投機が実効しない模様(未解決)。
運用: 既定は8B CPT Q8(鮮度優先)。32Bは`--llm-url`差し替えで品質優先モードとして利用可能。

### 結論と法則

1. **Swallow-8B-CPTの選択はデータで正当化された**: 自然さ系3指標すべてで最良(タイ含む)、
   hit@1も2番手グループ、速度も良好。総合最適。
2. **法則①: 日本語中心の事前学習が「自然さ」を決める** — Swallow CPT/llm-jp/sarashinaは
   逸脱率26〜28%帯、多言語Base(Qwen3 8B/14B)は32〜37%。素のQwen3-8B-Baseとの比較で
   SwallowのCPTの付加価値(逸脱-5.7pt、意図+6.8pt)が直接実証された。
3. **法則②: モデル容量は「字面一致」を決める** — hit@1は30B≈14B > 8B > 3B。
   ただし今回の30B/14BはCPT無しorQ4量子化のため自然さで負けた。
4. 将来の理想形は「30B級 × 日本語CPT × Q6以上 × 並列問題解消」。
   sarashina-3Bの224ms/自然さ健在も特筆(超低遅延が必要になった場合の控え)。
5. llm-jp-3-13b baseはconvert_hf_to_ggufのトークナイザ未対応で変換不可(instruct3で代替測定)。

## 2026-08-06 — モデルサイズ比較 (8B CPT vs 30B-A3B SFT/CPT)

「30Bにすれば予測は良くなるか」の決着実験。30B-A3B-CPTは既製GGUFが無いため
公式重み60GBを自前変換(convert_hf_to_gguf → Q4_K_M、`tts_server/convert_30b_cpt.sh`)。

| 指標 | **8B CPT Q8(現行)** | 30B-A3B SFT Q4 | 30B-A3B CPT Q4(自前変換) |
|---|---|---|---|
| hit@1 | 38.1% | 41.1% | 41.0% |
| hit@2 | 24.3% | 24.6% | **28.1%** |
| 逸脱率(judge) | **26.7%** | 30.3% | 31.5% |
| coherence平均 | **1.33** | 1.23 | 1.26 |
| 意図一致率 | **73.3%** | 69.1% | 68.5% |
| wall p50 (n=3) | ~500ms | 438ms | 1177ms※ |

※30B-CPT自前quantでは3並列が実質直列化(prompt 21ms/decode 283msなのにwall 1177ms)。
MoEのバッチ効率崩壊 or 量子化レイアウト差の疑い。未解決。

### 結論

- **30Bは字面一致(hit@1/@2)を確かに改善する**(+3〜4pt)が、ジャッジの自然さ指標では
  8B CPT Q8が上回り(量子化差 Q8 vs Q4 の交絡あり)、速度も現状は8Bが安定。
- → **本番は8B CPT Q8を継続**。
- 再挑戦の条件: ①Q6_K以上での再量子化(bf16から要再変換) ②並列直列化の解消
  (n=1運用 or llama.cpp側調査) ③judgeの人手校正後の再判定。hit@2 +4ptは
  実力向上の示唆であり、条件が揃えば逆転の余地あり。
- 副産物の教訓: パイプ(`| tail`)は終了コードを隠して`&&`チェーンを進める —
  重要ファイルのrmを含むチェーンでは絶対に使わない(bf16を一度消失させた)。

## 2026-08-04 — 文脈整合性評価系の導入 (LLM-judge + PPL-gap)

hit@k(字面一致)では測れない「流れに即しているか」を測る仕組みを追加した。

- **judge** (`scripts/judge_bench.py`): Swallow 8B **SFT** Q4_K_M(ジャッジ専用、port 18081、
  生成モデルとは別)が各候補を2軸で採点。coherence(文脈逸脱 0-2) / intent(実際の続きとの意図一致 0-2)。
  176行×2軸で約3分。※Qwen3系は `chat_template_kwargs: {enable_thinking: false}` 必須
- **PPL-gap** (`scripts/ppl_gap.py`): logP(候補|全文脈)−logP(候補|直近20字) をQwen3-0.6B-Base
  (CPU)で算出。「局所的には流暢だが文脈から浮く」候補の機械的検出。オンラインのゲート
  シグナルにも転用可能
- 長文脈fixture追加 (`fixtures/ja_long_sessions.jsonl`: 履歴200〜400字×5本)

### 結果(生成モデルはすべてSwallow 8B CPT Q8 @A6000、n=3)

| 指標 | 短文脈fixture(履歴0〜1文) | **長文脈fixture(履歴200〜400字)** |
|---|---|---|
| hit@1(字面一致) | 0.375 | 0.429 |
| **逸脱率(coherence=0)** | **0.318** | **0.071** |
| 意図一致率(intent>=1) | 0.676 | **0.893** |
| coherence平均 | 1.27 | 1.50 |
| wall p50 | 184ms | 218ms |

比較: Qwen3-4B instruct(短文脈): 逸脱率0.360 / 意図一致0.635 — Swallowと僅差だが
逸脱の質が悪い(「日本の教育制度」等、話題ごと飛ぶ)。

### 主要な発見

1. **文脈の効果は字面一致ではなく逸脱率に出る**: hit@1はほぼ横ばい(37.5→42.9%)だが、
   逸脱率は31.8%→7.1%(1/4以下)、意図一致は67.6%→89.3%。
   「短prefix+文脈なし」での逸脱は『このシステ→(システ)マティックレビュー』のような
   局所補完の暴走で、履歴があるとこのクラスがほぼ消える。
2. **hit@1は製品品質を大幅に過小評価**: hit@1=0の候補のうち57%はintent>=1(方向は同じ)、
   50%はcoherence=2(流れとして自然)。
3. **現状のconfidenceゲートは逸脱回避には弱い**: 上位50%に絞っても逸脱率31.8→27.3%止まり。
   逸脱を下げる第一手はゲート調整ではなく文脈を濃くすること。
4. PPL-gapは絶対閾値としては発火しない(生成が文脈条件付きのため)が、順位付けとして機能:
   最下位群に「(三秒長押しで)電源が切れます」(一般論としては自然だが本文脈では誤り)を
   正しく置いた。長文脈では平均+1.46で候補が文脈に強く支えられる。

### 注意(未校正)

- ジャッジは8B SFT単体・人手アンカー未実施。絶対値は±あり(「開催/開始」をcoherence=0と
  する等厳しめ)。同一ジャッジでの相対比較用として扱うこと。人手ラベル50〜100件での校正が
  次のステップ(UIの採用/拒否ログで日常的に溜める設計にする)。

### サーバ状態

- port 18080: 生成用 CPT Q8(`logs/llama.pid`) / port 18081: ジャッジ用 SFT Q4(`logs/judge.pid`)


## 2026-08-04 — Phase 2 A6000実測 (GPUサーバ, RTX A6000 48GB)

- モデル: **Qwen3-Swallow-8B-CPT-v0.2 Q8_0**(本命構成)。llama.cpp CUDA(sm_86)ビルド
- 場所: GPUサーバの `~/projects/PredictiveSpeaking`、llama-server port **18080**
  (8080はUtsushiMe系サービスが使用中のため専用ポート。PIDは`logs/llama.pid`)
- 併走: UtsushiMe stt-server / Irodori-TTS-Server / ollama と同居(GPU計~18GB/49GB)

| 設定 | hit@1 | hit@2 | speak率 | wall p50 | wall p95 | prompt_ms p50 | prompt_n中央値 |
|---|---|---|---|---|---|---|---|
| raw n=3 (server-local) | 0.375 | 0.233 | 0.98 | **183.9ms** | 233.1ms | 19.8ms | 3.0 |
| raw n=1 (server-local, 10件) | 0.375 | 0.214 | 0.93 | **143.1ms** | 147.1ms | 15.2ms | 2.0 |
| raw n=3 (Windows→LAN, 8件) | 0.283 | 0.152 | 0.96 | 194.8ms | 243.5ms | 16.5ms | 2.0 |

### 結論

1. **設計目標「候補更新200〜300ms」を実測で達成**(n=3で184ms、n=1なら143ms)。
   設計書の帯域ベース見積り(8B Q8で50〜65 tok/s)は実測とほぼ一致
   (predict 152ms/10tok×3並列 ≒ 65 tok/s/系統。A6000はバッチ3がほぼ無償)。
2. **LAN越しのオーバーヘッドは~10ms** — 「音声はWindows、LLMはA6000サーバ」の
   分離構成が成立する。
3. KV常駐はA6000でも機能(プリフィル中央値2〜3トークン、~20ms)。
4. 精度はhit@1=37.5%でQwen3-4B instructのraw(39.3%)と同水準。
   Swallow(日本語CPT)の優位は今回の30fixtureでは差が出ず → 精度改善は
   コンテキスト設計(history/原稿)とfixture拡充で追う(モデルサイズ比較も未実施)。
5. CPTモデル特有の現象: 空出力(empty 23/540候補)が一定数 → 改行等を
   最初に出して即stopするケース。restrictなsampling or 空時リトライで対処可能。

### サーバ運用メモ

- 起動: `~/projects/PredictiveSpeaking` で
  `LD_LIBRARY_PATH=llama.cpp/build/bin ./llama.cpp/build/bin/llama-server -m models/Qwen3-Swallow-8B-CPT-v0.2_Q8_0.gguf -ngl 99 --ctx-size 8192 --parallel 3 --host 0.0.0.0 --port 18080 --no-webui`
- **他サービスに注意**: 8080使用中・pkill禁止(自前PID `logs/llama.pid` のみkill)


## 2026-08-04 — Phase 2 予測ループ単体 (このWindowsマシン, Radeon 8060S / Vulkan)

- モデル: Qwen3-4B instruct Q4_K_M(手元にあったもの。本命のSwallow 8B CPTはA6000側で検証予定)
- fixtures: `fixtures/ja_utterances.jsonl` 30件 × 各6位置 = 180予測
- コマンド: `scripts/bench_predict.py --mode raw --n 3 --n-predict 10` ほか

| 設定 | hit@1 | hit@2 | hit@3 | speak率 | wall p50 | wall p95 | prompt_ms p50 | prompt_n中央値 |
|---|---|---|---|---|---|---|---|---|
| raw n=3 | **0.393** | 0.236 | 0.112 | 0.99 | 992ms | 1145ms | 69ms | **3.0** |
| raw n=1 (10件) | 0.407 | 0.222 | 0.074 | 0.90 | 521ms | 579ms | 63ms | 3.0 |
| chat (指示プロンプト) | 0.104 | 0.044 | 0.030 | 0.75 | 482ms | 827ms | — | — |

### 結論

1. **KV常駐方式は機能する**: `cache_prompt`+slot固定により毎サイクルのプリフィルは
   中央値3トークン(~70ms)。設計の核となる仮定が実証された。
2. **素の継続(raw)は指示プロンプト(chat)よりhit@1で約4倍正確**(39% vs 10%)。
   同一モデル・同一入力での比較であり、「baseモデルに素のテキスト継続をさせる」方針を
   強く支持する。chatモードはアシスタント応答混入(assistant_like/echo)も多い。
3. 遅延はiGPU(Vulkan)なりの値: 1候補デコード~22 tok/s、3並列で~12 tok/s/系統。
   n=3のwall p50=992ms。A6000(CUDA)では単発50〜65 tok/s級・並列コストほぼ無しの
   見込みで、**wall 200〜300ms(設計目標)はA6000側で検証する**。
4. flags分布(180件中): truncated 72 / question 17 / digits 16 / negation 15 /
   assistant_like 9 / echo 1。→ n_predict=10でも長すぎる出力が多く、
   truncationが正常系として機能している。assistant_like 9件はinstructモデル由来の
   汚染で、CPT(base)モデルでの再測定ポイント。

### 次のアクション

- [ ] A6000サーバでQwen3 Swallow 8B(CPT系)GGUFを用意し同ベンチ実行(READMEの手順)
- [ ] 精度レバー: history有無・preamble有無・n_predict 6〜8 の比較
- [ ] realtimeモード(PredictionLoopに時間刻みでpartialを流し込み、freshness_msと
      コアレス挙動を計測)
- [ ] hit@k の位置別分析(文節境界 vs 語中でのカット位置の影響)
