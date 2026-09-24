# models/

モデル・実行ファイルの置き場所(このREADME以外はgit管理外)。

```
uv run python scripts/setup_models.py     # 既定構成に必要なものを取得
```

| パス | 内容 | 取得方法 |
|---|---|---|
| `llm/llama-server(.exe)` + DLL | ローカルLLMサーバ(llama.cpp) | setup_models.py(Windowsは自動、他OSはPATHのllama-serverを使用) |
| `llm/qwen3-4b-q4_k_m.gguf` | 予測用LLM(Qwen3-4B, 2.5GB) | setup_models.py |
| `piper/<voice>.onnx` + `.onnx.json` | TTS。`en_US-lessac-medium`は市販声 | 市販声: setup_models.py / 本人声: `server/piper_train/` で学習 |
| `kokoro/` | 予備のローカルTTS(クローン不可) | `setup_models.py --kokoro` |
| `vosk-model-en-us-0.22-lgraph/` | 完全オフラインASR | `setup_models.py --vosk` |

本人声のPiperモデルは実在人物の声なので、公開リポジトリには含めず個別に受け渡すこと。
