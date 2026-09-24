"""起動済みアプリのE2Eスモーク(WebSocket経由で操作・検証する)。

先に音声なしモードでアプリを起動しておく:
  uv run python scripts/run_app.py --asr typed --port 8771
その後:
  uv run python scripts/smoke_e2e.py --part 2 --app-url http://127.0.0.1:8771   # タップ発話
  uv run python scripts/smoke_e2e.py --part 3 --app-url http://127.0.0.1:8771   # 自動発話

part 2: キャッシュ準備→ボタン押下(タップ)→再生発火→再タップで停止 を検証
part 3: 自動発話モードをONにし、ボタンなしで発火→停止 を検証
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def part2(app_url: str) -> None:
    import websockets

    uri = app_url.replace("http", "ws") + "/ws"
    async with websockets.connect(uri) as ws:
        fired_at = None
        played = False
        for i in range(600):  # 最大60秒
            snap = json.loads(await ws.recv())
            retry = fired_at is not None and i - fired_at > 30 and not snap["playing"]
            if snap.get("cache_matches") and (fired_at is None or retry):
                print(f"[ws] cache一致: '{snap['tts']['text']}' → press(タップ)")
                await ws.send(json.dumps({"type": "button", "pressed": True}))
                await ws.send(json.dumps({"type": "button", "pressed": False}))
                fired_at = i
            if fired_at is not None and snap["playing"]:
                print(f"[ws] PLAYING: '{snap['playing_text']}' gate_fired={snap['gate_fired']}")
                played = True
                # タップ方式: もう一度押す=キャンセル
                await ws.send(json.dumps({"type": "button", "pressed": True}))
                await ws.send(json.dumps({"type": "button", "pressed": False}))
                break
        if not played:
            print("[ws] FAIL: 再生が発火しなかった")
            sys.exit(1)
        for _ in range(20):
            snap = json.loads(await ws.recv())
            if not snap["playing"]:
                print("[ws] stopped OK")
                print("[ws] stats:", json.dumps(snap["stats"], ensure_ascii=False),
                      "synth_count:", snap["tts"]["synth_count"])
                return
        print("[ws] FAIL: 停止しなかった")
        sys.exit(1)


async def part3(app_url: str) -> None:
    import websockets

    uri = app_url.replace("http", "ws") + "/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "auto_mode", "value": True}))
        played = False
        for _ in range(900):  # 最大90秒
            snap = json.loads(await ws.recv())
            if snap["playing"] and not played:
                played = True
                print(f"[ws] AUTO-PLAYING: '{snap['playing_text']}' (無音{snap.get('silence_ms')}ms)")
            elif played and not snap["playing"]:
                print("[ws] stopped OK (バージインまたは再生完了)")
                return
        if not played:
            print("[ws] FAIL: 自動発火しなかった")
            sys.exit(1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-url", default="http://127.0.0.1:8770")
    ap.add_argument("--part", choices=["2", "3"], required=True)
    args = ap.parse_args()
    if args.part == "2":
        await part2(args.app_url)
    else:
        await part3(args.app_url)


if __name__ == "__main__":
    asyncio.run(main())
