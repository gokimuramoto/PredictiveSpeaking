"""ボイス切替のスモーク: 一覧取得→別ボイスへ切替→合成継続を確認。"""

from __future__ import annotations

import asyncio
import json
import sys


async def main(app_url: str = "http://127.0.0.1:8770") -> None:
    import websockets

    async with websockets.connect(app_url.replace("http", "ws") + "/ws") as ws:
        snap = json.loads(await ws.recv())
        print("voices:", snap.get("voices"), "active:", snap.get("active_voice"))
        if not snap.get("voices"):
            print("FAIL: ボイス一覧が空")
            sys.exit(1)
        target = next((v for v in snap["voices"] if v != snap.get("active_voice") and v != "none"), None)
        if not target:
            print("SKIP: 切替先なし")
            return
        await ws.send(json.dumps({"type": "set_voice", "voice": target}))
        synth_before = snap["tts"]["synth_count"]
        ok = False
        for _ in range(300):  # 30秒
            snap = json.loads(await ws.recv())
            if snap.get("active_voice") == target and snap["tts"]["synth_count"] > synth_before \
                    and not snap["tts"]["error"]:
                ok = True
                break
        print(f"switched to '{target}': {'OK — 切替後も合成が動作' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8770"))
