"""実行時設定の集約点(.env / 環境変数)。

個人環境のホスト名やパスをコードに焼き込まないため、環境依存の値はすべてここを経由する。
リポジトリ直下の .env を読む(.env.example 参照)。

  PS_SERVER_HOSTS  GPUサーバの候補ホスト(カンマ区切り・優先順)。日本語版やサーバ型TTSで使用
  PS_LLM_DIR       ローカルLLM(llama-server + gguf)の置き場所。既定: models/llm
  PS_LLM_MODEL     ローカルLLMのggufファイル名。既定: qwen3-4b-q4_k_m.gguf
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"

DEFAULT_LLM_MODEL = "qwen3-4b-q4_k_m.gguf"


def load_env() -> None:
    """リポジトリ直下の .env を読む(呼び出し元の場所に依存しない)。"""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")


def server_hosts() -> list[str]:
    """GPUサーバの候補ホスト(優先順)。例: PS_SERVER_HOSTS=192.168.0.10,my-server"""
    raw = os.environ.get("PS_SERVER_HOSTS", "127.0.0.1")
    return [h.strip() for h in raw.split(",") if h.strip()] or ["127.0.0.1"]


def server_host() -> str:
    """最優先のサーバホスト(ベンチ等、到達確認なしで使うスクリプト向け)。"""
    return server_hosts()[0]


def llm_dir() -> Path:
    p = Path(os.environ.get("PS_LLM_DIR", str(MODELS_DIR / "llm")))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def llm_model_path() -> Path:
    return llm_dir() / os.environ.get("PS_LLM_MODEL", DEFAULT_LLM_MODEL)


def llama_server_exe() -> Path | None:
    """llama-server の実行ファイル。PS_LLM_DIR内 → PATH の順に探す。"""
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    local = llm_dir() / name
    if local.exists():
        return local
    found = shutil.which("llama-server")
    return Path(found) if found else None
