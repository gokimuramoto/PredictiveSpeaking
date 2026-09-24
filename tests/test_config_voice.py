"""環境設定(config)とPiperの既定声選択のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config
from predictive_speaking.tts.piper_client import PiperTTSClient


def _touch_models(d: Path, names: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.onnx").write_bytes(b"")


def test_piper_prefers_own_voice_over_stock(tmp_path):
    _touch_models(tmp_path, ["en_US-lessac-medium", "en_US-zed-medium"])
    assert PiperTTSClient(model_dir=tmp_path).voice == "en_US-zed-medium"


def test_piper_falls_back_to_stock_voice(tmp_path):
    _touch_models(tmp_path, ["en_US-lessac-medium"])
    assert PiperTTSClient(model_dir=tmp_path).voice == "en_US-lessac-medium"


def test_piper_no_models_gives_empty_voice(tmp_path):
    assert PiperTTSClient(model_dir=tmp_path).voice == ""


def test_piper_explicit_voice_wins(tmp_path):
    _touch_models(tmp_path, ["en_US-lessac-medium", "en_US-zed-medium"])
    assert PiperTTSClient(model_dir=tmp_path, voice="en_US-lessac-medium").voice == "en_US-lessac-medium"


def test_server_hosts_parsing(monkeypatch):
    monkeypatch.setenv("PS_SERVER_HOSTS", " 10.0.0.5, my-server ,, ")
    assert config.server_hosts() == ["10.0.0.5", "my-server"]
    assert config.server_host() == "10.0.0.5"
    monkeypatch.delenv("PS_SERVER_HOSTS")
    assert config.server_hosts() == ["127.0.0.1"]


def test_llm_dir_relative_to_repo(monkeypatch):
    monkeypatch.setenv("PS_LLM_DIR", "models/llm")
    assert config.llm_dir() == (config.REPO_ROOT / "models" / "llm").resolve()
    monkeypatch.setenv("PS_LLM_MODEL", "x.gguf")
    assert config.llm_model_path().name == "x.gguf"
