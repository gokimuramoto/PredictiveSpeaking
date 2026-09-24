"""候補音声の先行合成キャッシュ(SpeechPrefetcher)。

予測ループの最良候補が変わるたびにTTSを先行実行し、PCMをキャッシュする。
ゲートが開いた瞬間はキャッシュ済みPCMを解放するだけ、が設計の核。

- 候補テキストが同じ間は再合成しない
- 新候補が来たら進行中の合成をキャンセルして最新だけ合成(latest-value)
- 鮮度: 生成元入力(input_latest)と生成時刻を保持し、ゲート側が判断に使う
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..clock import now_ms
from ..events import EventLogger
from ..predict.types import CandidateSet
from ..text import ja


@dataclass
class CachedAudio:
    text: str
    pcm_s16le: bytes
    sample_rate: int
    source_input: str        # この候補の由来partial
    created_t_ms: float
    ttfa_ms: float
    synth_ms: float

    @property
    def duration_ms(self) -> float:
        return len(self.pcm_s16le) / 2 / self.sample_rate * 1000

    def age_ms(self) -> float:
        return now_ms() - self.created_t_ms


class SpeechPrefetcher:
    def __init__(self, tts_client, logger: EventLogger | None = None,
                 min_chars: int = 3, enabled: bool = True):
        self.tts = tts_client
        self.log = logger or EventLogger(None)
        self.min_chars = min_chars
        self.enabled = enabled
        self.current: CachedAudio | None = None
        self.last_error: str | None = None
        self.synth_count = 0
        # コアレス方式: 合成は常に1本だけ。実行中に来た新候補は「最新の希望」として
        # 上書きし、完了時に希望と違えば取り直す。キャンセルはしない(サーバ側の
        # 合成は止まらず、直列サーバではキュー滞留の原因になるだけ)。
        self._desired: tuple[str, str] | None = None  # (text, source_input)
        self._pinned: tuple[str, str] | None = None
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None

    # ---- ピン留め: 「この候補を最優先で合成せよ」(ボタン押下時に使う) ----
    def request_pin(self, text: str, source_input: str) -> None:
        self._pinned = (text, source_input)
        if self.current and self.current.text == text:
            return  # 既に音声がある
        self._desired = (text, source_input)
        self._wake.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    def clear_pin(self) -> None:
        self._pinned = None

    # PredictionLoopのon_resultから呼ぶ(同期)
    def on_result(self, result: CandidateSet) -> None:
        if not self.enabled:
            return
        if getattr(self, "_pinned", None):
            return  # ピン留め中は新候補で上書きしない(押した瞬間の候補を最優先)
        best = result.best
        if not best or not best.speakable or len(best.text) < self.min_chars:
            return
        text = best.text
        if self.current and self.current.text == text:
            # 同一テキスト: 由来だけ更新して鮮度を保つ
            self.current.source_input = result.input_latest
            self.current.created_t_ms = now_ms()
            return
        self._desired = (text, result.input_latest)
        self._wake.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            desired = self._desired
            if desired is None:
                return
            text, source_input = desired
            if self.current and self.current.text == text:
                continue
            t0 = now_ms()
            try:
                pcm, ttfa = await self.tts.synthesize(text)
            except Exception as e:
                self.last_error = str(e)
                self.log.log("tts_error", text=text, error=str(e)[:200])
                await asyncio.sleep(0.3)
                continue
            self.synth_count += 1
            self.last_error = None
            # 完成した合成は常にキャッシュする(合成開始時点では最良候補だったもの)。
            # 「今の発話に対して再生してよいか」は fresh() が再生時に判定するので、
            # ここで破棄すると候補の変化が速いときにキャッシュが永遠に埋まらない。
            self.current = CachedAudio(
                text=text, pcm_s16le=pcm, sample_rate=self.tts.sample_rate,
                source_input=source_input, created_t_ms=now_ms(),
                ttfa_ms=ttfa, synth_ms=now_ms() - t0,
            )
            self.log.log("tts_cached", text=text, synth_ms=round(now_ms() - t0, 1),
                         ttfa_ms=round(ttfa, 1),
                         duration_ms=round(self.current.duration_ms, 1))
            if self._desired and self._desired[0] != text:
                self._wake.set()  # より新しい希望が来ているので続けて取りに行く

    def invalidate(self) -> None:
        """発話が確定(final)したら呼ぶ。継続点が消えた候補は無効。"""
        self.current = None
        self._desired = None

    def fresh(self, current_partial: str, ttl_ms: float = 6000.0) -> CachedAudio | None:
        """再生に使える鮮度のキャッシュを返す。

        条件: 現在発話中(partial非空)で、TTL内、かつ候補の由来partialが
        現在partialと整合すること。比較は正規化(句読点・空白無視)で行う —
        Azureはpartialを微修正(読点挿入等)するため厳密一致では誤ブロックする。
        発話をまたいだ古い候補はinvalidate()で消える。
        """
        c = self.current
        if c is None or not current_partial or c.age_ms() > ttl_ms:
            return None
        np_ = ja.normalize_for_match(current_partial)
        ns = ja.normalize_for_match(c.source_input)
        if not np_.startswith(ns[: len(np_)]) and not ns.startswith(np_):
            return None
        return c
