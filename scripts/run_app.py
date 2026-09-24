"""PredictiveSpeaking 本体: マイク→ASR→常時予測→TTS先行合成→ゲート再生 + Web UI。

既定構成(引数なし): 英語 / ASR=Azure(クラウド) / LLM=このPCのllama-serverを自動起動 /
TTS=Piper(本人声モデル, このPCのCPU)。起動後 http://127.0.0.1:8770 を開く。

例:
  uv run python scripts/run_app.py                          # 既定構成
  uv run python scripts/run_app.py --voice en_US-muramoto-medium
  uv run python scripts/run_app.py --asr typed --no-tts     # 音声なしの配管テスト
  uv run python scripts/run_app.py --lang ja                # 日本語版(GPUサーバ構成, .envのPS_SERVER_HOSTS)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking import config  # noqa: E402
from predictive_speaking.audio.playback import PlaybackController  # noqa: E402
from predictive_speaking.events import EventLogger  # noqa: E402
from predictive_speaking.gate import Gate  # noqa: E402
from predictive_speaking.predict.llamacpp import LlamaCppPredictor  # noqa: E402
from predictive_speaking.predict.loop import PredictionLoop  # noqa: E402
from predictive_speaking.session import LiveSession  # noqa: E402
from predictive_speaking.tts.cache import SpeechPrefetcher  # noqa: E402
from predictive_speaking.webapp import AppState, create_app  # noqa: E402

SERVER_TTS_PORTS = {"irodori": 8088, "chatterbox": 18083, "chatterbox-vllm": 18084}
SERVER_LLM_PORTS = {"ja": 18080, "en": 18082}


async def main() -> None:
    ap = argparse.ArgumentParser(description="PredictiveSpeaking (常時発話予測+本人声補助)")
    ap.add_argument("--lang", choices=["en", "ja"], default="en")
    ap.add_argument("--asr", choices=["azure", "vosk", "typed"], default="azure",
                    help="azure=クラウド(既定・実声品質最良) / vosk=完全オフライン / typed=文字送りテスト")
    ap.add_argument("--llm", choices=["local", "server"], default=None,
                    help="既定: en=local(このPCでllama-serverを自動起動) / ja=server")
    ap.add_argument("--llm-url", default=None, help="LLMのURLを直接指定(llama-server互換)")
    ap.add_argument("--tts-backend",
                    choices=["piper", "kokoro", "chatterbox", "chatterbox-vllm", "irodori"],
                    default=None,
                    help="既定: en=piper(本人声・ローカル) / ja=irodori(サーバ)")
    ap.add_argument("--tts-url", default=None, help="サーバ型TTSのURLを直接指定")
    ap.add_argument("--voice", default=None,
                    help="声。省略時: piperは前回UIで選んだ声(なければ先頭のモデル)")
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="予測候補数(既定: ローカルLLM=1 / サーバLLM=3)")
    ap.add_argument("--local-llm-port", type=int, default=18085)
    ap.add_argument("--local", action="store_true", help=argparse.SUPPRESS)  # 旧フラグ互換(既定化済み)
    ap.add_argument("--mic-device", default=None,
                    help="入力デバイス(番号 or 名前の一部)。省略時はOSの既定入力。"
                         "一覧: uv run python -c \"import sounddevice;print(sounddevice.query_devices())\"")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--log", default=None, help="省略時は logs/session_<日時>.jsonl")
    ap.add_argument("--context-file", default=None, help="話題メモ(原稿)のテキストファイル")
    args = ap.parse_args()

    # 相対パスは呼び出し場所ではなくリポジトリ直下基準で解決する
    if args.context_file:
        args.context_file = str(Path(args.context_file).resolve())
    if args.log:
        args.log = str(Path(args.log).resolve())
    os.chdir(config.REPO_ROOT)
    config.load_env()

    if args.log is None:
        from datetime import datetime

        args.log = f"logs/session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    logger = EventLogger(args.log)

    # ---- 構成の解決 ----
    if args.local:
        args.llm = "local"
    if args.llm_url:
        args.llm = "server"
    if args.llm is None:
        args.llm = "local" if args.lang == "en" else "server"
    if args.tts_backend is None:
        args.tts_backend = "piper" if args.lang == "en" else "irodori"
    if args.voice is None:
        if args.tts_backend == "piper":
            saved = _load_saved_voice()
            args.voice = saved if saved and (config.MODELS_DIR / "piper" / f"{saved}.onnx").exists() else ""
        else:
            args.voice = {"kokoro": "af_heart", "chatterbox": "default",
                          "chatterbox-vllm": "default"}.get(args.tts_backend, "me")
    local_llm = args.llm == "local"
    if args.n_candidates is None:
        args.n_candidates = 1 if local_llm else 3
    print(f"[config] lang={args.lang} asr={args.asr} llm={args.llm} tts={args.tts_backend}")

    import socket as _socket

    def _reachable(host: str, port: int, timeout: float) -> bool:
        try:
            with _socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    # ローカルLLM: 起動していなければllama-serverを自動起動する
    local_llm_proc = None
    if local_llm:
        args.llm_url = f"http://127.0.0.1:{args.local_llm_port}"
        if not _reachable("127.0.0.1", args.local_llm_port, timeout=0.3):
            import subprocess

            exe, model = config.llama_server_exe(), config.llm_model_path()
            if exe is None or not model.exists():
                print(f"[llm] エラー: ローカルLLMが見つかりません (llama-server={exe}, model={model})")
                print("      → uv run python scripts/setup_models.py で取得してください")
                return
            print(f"[llm] ローカルllama-server起動中: {model.name} ...")
            local_llm_proc = subprocess.Popen(
                [str(exe), "-m", str(model), "-ngl", "99", "--ctx-size", "8192",
                 "--parallel", "3", "--port", str(args.local_llm_port),
                 "--host", "127.0.0.1", "--no-webui"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(120):
                if _reachable("127.0.0.1", args.local_llm_port, timeout=0.5):
                    break
                if local_llm_proc.poll() is not None:
                    print("[llm] エラー: llama-serverが起動に失敗しました")
                    return
                await asyncio.sleep(1.0)
            print(f"[llm] ローカルLLM準備完了 (port {args.local_llm_port})")

    # GPUサーバを使う構成のときだけ、.envの候補ホストを順に到達確認する
    # (Tailscale等はアイドルからの初回接続に数秒かかるため、タイムアウトを伸ばして二巡)
    hosts = config.server_hosts()
    backend_host = hosts[0]
    need_server_llm = args.llm_url is None
    need_server_tts = args.tts_backend in SERVER_TTS_PORTS and args.tts_url is None
    if need_server_llm or need_server_tts:
        probe_port = SERVER_LLM_PORTS[args.lang] if need_server_llm else SERVER_TTS_PORTS[args.tts_backend]
        found = None
        for to in (0.7, 3.0):
            found = next((h for h in hosts if _reachable(h, probe_port, timeout=to)), None)
            if found:
                break
        if found is None:
            print(f"[net] 警告: GPUサーバに届きません (PS_SERVER_HOSTS={','.join(hosts)}, port {probe_port})")
        else:
            backend_host = found
            if found != hosts[0]:
                print(f"[net] {hosts[0]} 不達 → {found} 経由で接続します")
    if args.llm_url is None:
        args.llm_url = f"http://{backend_host}:{SERVER_LLM_PORTS[args.lang]}"

    # ローカルLLMではn_probs=0が必須級: iGPU等ではlogprobs抽出が24トークンあたり~1秒
    # かかる(GPUサーバでは無視できるコストのため有効のまま)
    predictor = LlamaCppPredictor(base_url=args.llm_url, mode="raw",
                                  n_candidates=args.n_candidates,
                                  n_predict=24, lang=args.lang,
                                  n_probs=0 if local_llm else 1)
    if args.tts_backend == "kokoro":
        from predictive_speaking.tts.kokoro_client import KokoroTTSClient

        tts = KokoroTTSClient(voice=args.voice)
    elif args.tts_backend == "piper":
        from predictive_speaking.tts.piper_client import PiperTTSClient

        tts = PiperTTSClient(voice=args.voice)  # voice=""なら models/piper の先頭モデル
        if not tts.voice:
            print("[tts] エラー: Piperモデルがありません → uv run python scripts/setup_models.py")
            return
        print(f"[tts] piper voice: {tts.voice}")
    else:
        from predictive_speaking.tts.irodori_client import IrodoriTTSClient

        port = SERVER_TTS_PORTS[args.tts_backend]
        tts = IrodoriTTSClient(base_url=args.tts_url or f"http://{backend_host}:{port}",
                               voice=args.voice)
    prefetcher = SpeechPrefetcher(tts, logger=logger, enabled=not args.no_tts)

    def on_result(result) -> None:
        prefetcher.on_result(result)

    loop = PredictionLoop(predictor, on_result=on_result, logger=logger)

    if args.asr == "typed":
        from predictive_speaking.asr.typed import TypedASR

        script = []
        if args.lang == "en":
            import re as _re

            with open("fixtures/en_utterances.jsonl", encoding="utf-8") as fh:
                for line in list(fh)[:6]:
                    item = json.loads(line)
                    full = (item.get("history", "") + " " + item["text"]).strip()
                    script.extend([s.strip() for s in _re.split(r"(?<=[.!?])\s+", full) if s.strip()])
        else:
            with open("fixtures/ja_long_sessions.jsonl", encoding="utf-8") as fh:
                for line in list(fh)[:2]:
                    item = json.loads(line)
                    script.extend([s for s in (item.get("history", "") + item["text"]).split("。") if s])
        asr = TypedASR(script, tick_ms=400, sentence_gap_ms=3000, stall_ms=3000)
    elif args.asr == "vosk":
        from predictive_speaking.asr.vosk_asr import VoskASR

        asr = VoskASR(lang=args.lang)
    else:
        from predictive_speaking.asr.azure_speech import AzureSpeechASR

        # en: 確定(ピリオド)が来るとpartialが空になり自動発話の窓が閉じるため、
        # 確定までの無音を長めに取る(猶予ウィンドウと併せた二段構え)
        asr = AzureSpeechASR(language="en-US" if args.lang == "en" else "ja-JP",
                             segmentation_silence_ms=4500 if args.lang == "en" else 2800)

    session = LiveSession(asr, loop, logger=logger,
                          on_final=lambda _text: prefetcher.invalidate(),
                          lang=args.lang)
    playback = PlaybackController(sample_rate=tts.sample_rate)
    gate = Gate()
    state = AppState(session, loop, prefetcher, playback, gate, logger,
                     predictor=predictor, lang=args.lang,
                     ui_state_path="logs/ui_state.json")
    if args.context_file:
        state.set_context(Path(args.context_file).read_text(encoding="utf-8"))
        print(f"[context] {len(state.context_text)}文字の話題メモを注入")
    app = create_app(state)

    # 起動
    if not args.no_tts:
        # サーバ型TTSの一時不達(Tailscaleのコールドスタート等)に備え数回リトライする
        last_err = None
        for attempt in range(3):
            try:
                health = await tts.health()
                print(f"[tts] {health}")
                if hasattr(tts, "probe"):
                    rate = await tts.probe()
                    playback.sample_rate = rate
                    print(f"[tts] sample_rate={rate}")
                await state.refresh_voices()
                print(f"[tts] voices: {state.voices} (active: {state.active_voice})")
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    print(f"[tts] 接続リトライ中... ({e})")
                    await asyncio.sleep(2.5)
        if last_err is not None:
            print(f"[tts] 警告: TTSを初期化できません ({last_err}) — 先行合成は無効")
            prefetcher.enabled = False
    await session.start()

    mic = None
    if args.asr in ("azure", "vosk"):
        from predictive_speaking.audio.capture import MicSource

        # デバイス指定: 番号そのまま / 文字列なら入力デバイス名の部分一致で解決
        dev = None
        try:
            import sounddevice as _sd

            if args.mic_device is not None:
                if args.mic_device.isdigit():
                    dev = int(args.mic_device)
                else:
                    key = args.mic_device.lower()
                    for i, d in enumerate(_sd.query_devices()):
                        if d["max_input_channels"] > 0 and key in d["name"].lower():
                            dev = i
                            break
                    if dev is None:
                        print(f"[mic] 警告: '{args.mic_device}' に一致する入力デバイスなし(既定を使用)")
            _dev = _sd.query_devices(dev if dev is not None else _sd.default.device[0])
            # 起動時にどのデバイスを掴んだか明示する(取り違え・無音デバイス事故の早期発見)
            print(f"[mic] 入力デバイス: {_dev['name']}")
        except Exception as e:
            print(f"[mic] 入力デバイス情報を取得できません: {e}")
        mic = MicSource(device=dev)
        # 半二重・声登録録音を含むマイク処理はAppStateに一元化
        mic.start(state.feed_mic)
        state.mic = mic
    playback.start()

    async def control_tick() -> None:
        while True:
            state.tick()
            await asyncio.sleep(0.05)  # 20Hz

    tick_task = asyncio.create_task(control_tick())

    import socket

    import uvicorn

    # 旧インスタンスがポートを握っていたら引き継ぎ(shutdown要求→解放待ち)。
    # bind失敗のまま旧コードのUIに繋がる事故を根治する。
    import httpx as _httpx

    for attempt in range(10):
        try:
            probe = socket.socket()
            probe.bind(("127.0.0.1", args.port))
            probe.close()
            break
        except OSError:
            if attempt == 0:
                print(f"[app] ポート{args.port}使用中 → 旧インスタンスに終了を要求します")
                try:
                    _httpx.post(f"http://127.0.0.1:{args.port}/shutdown", timeout=3)
                except Exception:
                    pass
            elif attempt == 5 and sys.platform == "win32":
                # 旧コード(/shutdown未対応)の残骸はOSレベルで解放する
                print(f"[app] 応答なし → ポート{args.port}の保持プロセスを強制終了します")
                import subprocess

                subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"Get-NetTCPConnection -LocalPort {args.port} -State Listen -ErrorAction SilentlyContinue | "
                     "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"],
                    capture_output=True, timeout=15)
            await asyncio.sleep(0.5)
    else:
        print(f"[app] エラー: ポート{args.port}を解放できません。旧プロセスを手動終了してください")
        return

    uv_config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(uv_config)
    app.state.shutdown_cb = lambda: setattr(server, "should_exit", True)
    print(f"\n🎙  PredictiveSpeaking: http://127.0.0.1:{args.port}  (Ctrl+C で終了)")
    print(f"[log] {args.log}\n")
    try:
        await server.serve()
    finally:
        tick_task.cancel()
        if mic:
            mic.stop()
        playback.close()
        await session.stop()
        await predictor.close()
        await tts.close()
        if local_llm_proc is not None and local_llm_proc.poll() is None:
            local_llm_proc.terminate()  # 自分で起動したローカルLLMは自分で片付ける
        logger.close()


def _load_saved_voice() -> str | None:
    """前回UIで選んだ声(logs/ui_state.json)。"""
    try:
        return json.loads((config.LOGS_DIR / "ui_state.json").read_text(encoding="utf-8")).get("voice")
    except Exception:
        return None


if __name__ == "__main__":
    asyncio.run(main())
