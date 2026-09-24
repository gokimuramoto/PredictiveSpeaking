"""既定構成(ローカルLLM + Piper TTS)に必要なモデル・実行ファイルを models/ に揃える。

  uv run python scripts/setup_models.py              # 既定: LLM(llama.cpp + Qwen3-4B) + Piper市販声
  uv run python scripts/setup_models.py --kokoro     # 予備のローカルTTS(Kokoro, ~340MB)も取得
  uv run python scripts/setup_models.py --vosk       # 完全オフラインASR(Vosk英語, ~130MB)も取得

既に存在するファイルはスキップする(何度実行してもよい)。
本人声のPiperモデル(<voice>.onnx + .onnx.json)は server/piper_train/ で学習して models/piper/ に置く。

llama-server は Windows では Vulkan 版(GPU/iGPU対応)を自動取得する。
Linux/macOS では llama.cpp をインストールして llama-server を PATH に通すこと
(または .env の PS_LLM_DIR にバイナリの場所を指定)。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config  # noqa: E402

QWEN_GGUF_URL = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOSK_EN_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"


def download(url: str, dest: Path, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {label}: {dest.relative_to(config.REPO_ROOT)} (既存)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [dl] {label}: {done / total:5.1%} of {total / 1e6:.0f}MB", end="", flush=True)
    tmp.replace(dest)
    print(f"\r  [ok] {label}: {dest.relative_to(config.REPO_ROOT)} ({dest.stat().st_size / 1e6:.0f}MB)      ")


def setup_llama_server() -> None:
    exe = config.llama_server_exe()
    if exe is not None:
        print(f"  [skip] llama-server: {exe} (既存)")
        return
    if sys.platform != "win32":
        print("  [要手動] llama-server が見つかりません。llama.cpp をインストールして PATH に通してください")
        print("           (例: brew install llama.cpp / ソースからビルド)")
        return
    # 直近のリリースから Windows Vulkan 版を探す(latestタグが別物を指すことがあるため)
    releases = httpx.get(LLAMA_RELEASES_API, timeout=30.0, follow_redirects=True).json()
    url = None
    for rel in releases:
        for asset in rel.get("assets", []):
            if re.fullmatch(r"llama-b\d+-bin-win-vulkan-x64\.zip", asset.get("name", "")):
                url = asset["browser_download_url"]
                break
        if url:
            break
    if url is None:
        print("  [要手動] llama.cpp の Windows Vulkan ビルドが見つかりません。"
              "https://github.com/ggml-org/llama.cpp/releases から取得して models/llm/ に展開してください")
        return
    print(f"  [dl] llama.cpp: {url.rsplit('/', 1)[-1]}")
    data = httpx.get(url, timeout=300.0, follow_redirects=True).content
    dest = config.llm_dir()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            # llama-server の実行に必要なもの(本体 + DLL)だけ展開する
            if base == "llama-server.exe" or base.endswith(".dll"):
                (dest / base).write_bytes(zf.read(name))
    print(f"  [ok] llama-server: {dest / 'llama-server.exe'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="モデル・実行ファイルのセットアップ")
    ap.add_argument("--kokoro", action="store_true", help="予備のローカルTTS(Kokoro)も取得")
    ap.add_argument("--vosk", action="store_true", help="完全オフラインASR(Vosk英語)も取得")
    ap.add_argument("--no-llm", action="store_true", help="LLMの取得を省く")
    args = ap.parse_args()
    config.load_env()

    if not args.no_llm:
        print("■ ローカルLLM (予測)")
        setup_llama_server()
        download(QWEN_GGUF_URL, config.llm_model_path(), "Qwen3-4B Q4_K_M")

    print("■ Piper TTS (市販の英語声: 本人声モデルが無い場合の既定)")
    piper_dir = config.MODELS_DIR / "piper"
    for name in ("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"):
        download(PIPER_BASE + name, piper_dir / name, name)
    own = sorted(p.stem for p in piper_dir.glob("*.onnx") if "lessac" not in p.stem)
    print(f"  本人声モデル: {', '.join(own) if own else '(なし — server/piper_train/ で作成できます)'}")

    if args.kokoro:
        print("■ Kokoro TTS (予備)")
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            download(KOKORO_BASE + name, config.MODELS_DIR / "kokoro" / name, name)

    if args.vosk:
        print("■ Vosk 英語 (完全オフラインASR)")
        target = config.MODELS_DIR / "vosk-model-en-us-0.22-lgraph"
        if target.exists():
            print(f"  [skip] Vosk: {target.relative_to(config.REPO_ROOT)} (既存)")
        else:
            zpath = config.MODELS_DIR / "vosk-en.zip"
            download(VOSK_EN_URL, zpath, "vosk-model-en-us-0.22-lgraph.zip")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(config.MODELS_DIR)
            zpath.unlink()
            print(f"  [ok] Vosk: {target.relative_to(config.REPO_ROOT)}")

    print("\n完了。起動: uv run python scripts/run_app.py")


if __name__ == "__main__":
    main()
