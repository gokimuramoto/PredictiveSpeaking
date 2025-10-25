# EchoNext - Digital Twin Voice System

リアルタイム音声継続システム。ユーザーの声をクローンし、GPT-4.1-miniで次の単語を予測して音声合成します。

## システム構成

- **ASR (音声認識)**: Browser Web Speech API
- **LLM (予測)**: Azure OpenAI GPT-4.1-mini
- **RAG (知識ベース)**: Azure OpenAI text-embedding-3-small + ベクトル検索（オプション）
- **TTS (音声合成)**: Cartesia API
- **テキスト修正**: Azure OpenAI GPT-4.1-mini

## 必要な環境

- Node.js 18以降
- Chrome または Edge ブラウザ（Web Speech API対応）
- Azure OpenAI API キー
- Cartesia API キー

## セットアップ手順

### 1. 依存パッケージのインストール

```bash
npm install
```

### 2. 環境変数の設定

`.env.example` を `.env` にコピーして、APIキーを設定してください。

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```env
# Cartesia API
CARTESIA_API_KEY=your_cartesia_api_key_here

# Azure OpenAI (gpt-4.1-mini)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-06-01

# Azure OpenAI Embedding (RAG用 - オプション)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Server Config
PORT=3000

# TTS Provider Selection
TTS_PROVIDER=cartesia

# ASR Provider Selection
ASR_PROVIDER=browser
```

### 3. サーバーの起動

```bash
npm start
```

サーバーが `http://localhost:3000` で起動します。

### 4. ブラウザでアクセス

Chrome または Edge で `http://localhost:3000` を開きます。

## 使い方

### 1. 言語選択フェーズ

1. 使用する言語（日本語または英語）を選択

### 2. 知識モデル選択フェーズ（オプション）

RAG知識ベースを使用すると、特定のドメイン知識に基づいた高精度な予測が可能になります。

#### オプション1: 既存のRAG知識ベースを読み込む

1. ドロップダウンから既存のRAGモデルを選択
2. 「RAGを読み込む」ボタンをクリック

#### オプション2: 新しいRAG知識ベースを作成（推奨）

1. プロジェクトルートの`knowledge-data/` フォルダ内に、ドメイン名でフォルダを作成
   ```bash
   mkdir knowledge-data/my-domain
   ```
2. 作成したフォルダに知識データファイルを配置
   - 対応形式: `.txt`, `.pdf`, `.docx`, `.tex`
   - 例: `knowledge-data/my-domain/document1.pdf`
3. ブラウザのドロップダウンからフォルダを選択
4. 知識データの主要言語を選択（日本語/英語）
5. 「RAG知識ベースを作成」ボタンをクリック
6. エンベディング生成が完了するまで待つ（数分かかる場合があります）
   - 生成されたRAG知識ベースは `rag-knowledge/my-domain.json` に保存されます

**注意**: RAG知識ベースの作成にはAzure OpenAI Embedding API（text-embedding-3-small）を使用するため、API使用料が発生します。

#### オプション3: モデルをスキップ

RAG知識ベースを使用せず、LLMのみで予測を行います。

### 3. セットアップフェーズ

1. 「録音開始」ボタンをクリック
2. 自己紹介など、10秒以上話す
3. 「録音停止」ボタンをクリック
4. 音声クローンが完了すると、メインフェーズに移行

### 4. メインフェーズ

1. 「システム開始」ボタンをクリック
2. 話すと、リアルタイムで次の単語が予測され、音声で再生されます
3. 予測履歴（直近3件）が表示されます
4. 「システム停止」で停止
5. 「リセット」でセットアップからやり直し

## RAG知識ベースについて

### 予測戦略

システムは以下の順序で予測を行います：

1. **RAG読み込み済み**: LLM + 知識ベース（ベクトル検索でコサイン類似度トップ3のチャンクを取得してLLMに渡す）
2. **RAG未読み込み**: 純粋なLLMのみで予測

### RAG知識ベースの仕組み

1. **テキストチャンキング**: ドキュメントを500文字のチャンクに分割（50文字のオーバーラップ）
2. **言語別の文章分割**:
   - 日本語: `。！？` で分割
   - 英語: `.!?` で分割
3. **エンベディング生成**: Azure OpenAI text-embedding-3-small（1536次元）
4. **ベクトル検索**: 入力テキストとコサイン類似度でトップ3チャンクを取得
5. **LLMコンテキスト**: 関連チャンクをプロンプトに含めてGPT-4.1-miniで予測

### ディレクトリ構造

```
knowledge-data/          # 知識データ配置フォルダ
├── my-domain/          # ドメイン別フォルダ（フォルダ名がモデル名になる）
│   ├── document1.pdf
│   ├── document2.txt
│   └── paper.tex
└── another-domain/
    └── data.docx

rag-knowledge/          # 生成されたRAG知識ベース（自動作成）
├── my-domain.json      # ベクトルDBファイル
└── another-domain.json
```

## ファイル構成

```
echonext/
├── backend/
│   ├── server.js              # メインサーバー（WebSocket + REST API）
│   ├── azurePredictor.js      # GPT-4.1-mini予測エンジン
│   ├── ragPredictor.js        # RAG予測エンジン（ベクトル検索）
│   ├── buildRAG.js            # RAG知識ベース構築スクリプト
│   ├── cartesiaTTS.js         # Cartesia音声合成
│   └── transcriptCorrector.js # テキスト修正
├── frontend/
│   ├── index.html             # メインHTML（多言語対応UI）
│   ├── app.js                 # フロントエンドロジック
│   └── style.css              # スタイル
├── knowledge-data/            # 知識データ配置フォルダ（ユーザー作成）
│   └── (your-domain)/         # ドメイン別フォルダ
├── rag-knowledge/             # 生成されたRAG知識ベース（自動作成）
│   └── *.json                 # ベクトルDBファイル
├── package.json
├── package-lock.json
├── .env.example               # 環境変数テンプレート
└── README.md                  # このファイル
```

## トラブルシューティング

### マイクが認識されない

- ブラウザの設定でマイクのアクセス許可を確認してください
- Chrome/Edgeを使用していることを確認してください

### 音声クローンが失敗する

- 10秒以上話していることを確認してください
- Cartesia APIキーが正しいことを確認してください
- 音声が明瞭に録音されていることを確認してください

### 予測が動作しない

- Azure OpenAI APIキーが正しいことを確認してください
- デプロイメント名が `gpt-4.1-mini` であることを確認してください

### RAG知識ベースの作成が失敗する

- `knowledge-data/` フォルダが存在し、その中にフォルダが作成されていることを確認してください
- フォルダ内に対応ファイル（.txt, .pdf, .docx, .tex）が存在することを確認してください
- Azure OpenAI Embedding APIキーが設定されていることを確認してください
- Embedding デプロイメント名が `text-embedding-3-small` であることを確認してください
- サーバーコンソールのログでエラー詳細を確認してください

### RAG予測の精度が低い

- 知識データの内容が予測したいドメインと一致しているか確認してください
- 知識データの言語設定（日本語/英語）が正しいか確認してください
- より多くの関連ドキュメントを追加してRAG知識ベースを再構築してください

## ライセンス

このプロジェクトは個人使用のみを想定しています。
