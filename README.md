# EchoNext - Digital Twin Voice System

リアルタイム音声継続システム。ユーザーの声をクローンし、GPT-4.1-miniで次の単語を予測して音声合成します。

## システム構成

- **ASR (音声認識)**: Browser Web Speech API
- **LLM (予測)**: Azure OpenAI GPT-4.1-mini
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

### 1. セットアップフェーズ

1. 「録音開始」ボタンをクリック
2. 自己紹介など、10秒以上話す
3. 「録音停止」ボタンをクリック
4. 音声クローンが完了すると、メインフェーズに移行

### 2. メインフェーズ

1. 「システム開始」ボタンをクリック
2. 話すと、リアルタイムで次の単語が予測され、音声で再生されます
3. 「システム停止」で停止
4. 「リセット」でセットアップからやり直し

## ファイル構成

```
echonext-clean/
├── backend/
│   ├── server.js              # メインサーバー
│   ├── azurePredictor.js      # GPT-4.1-mini予測エンジン
│   ├── cartesiaTTS.js         # Cartesia音声合成
│   ├── transcriptCorrector.js # テキスト修正
├── frontend/
│   ├── index.html             # メインHTML
│   ├── app.js                 # フロントエンドロジック
│   └── style.css              # スタイル
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

## ライセンス

このプロジェクトは個人使用のみを想定しています。
