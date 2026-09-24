"""LiveSession: 音声入力→ASR→stable prefix→常時予測ループの結線。

- ASR partialごとにStablePrefixTrackerを更新し、should_predictならループへ最新値を投入
- ASR finalで発話履歴に追加しトラッカーをリセット
- 状態スナップショットをUI(コンソール/将来のWeb)へ提供
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .asr.base import ASREvent, StreamingASR
from .clock import now_ms
from .events import EventLogger
from .text import ja
from .predict.loop import PredictionLoop
from .predict.types import CandidateSet
from .text.stable_prefix import StablePrefixTracker


@dataclass
class SessionState:
    history: list[str] = field(default_factory=list)
    partial: str = ""
    stable: str = ""
    last_result: CandidateSet | None = None
    asr_events: int = 0
    asr_intervals_ms: list[float] = field(default_factory=list)
    started_t_ms: float = 0.0
    last_activity_t_ms: float = 0.0  # 発話が最後に「進んだ」時刻(自動発話の無音判定に使う)
    last_final_text: str = ""        # 直近の確定発話(確定直後の自動発話猶予に使う)
    last_final_t_ms: float = 0.0


class LiveSession:
    def __init__(
        self,
        asr: StreamingASR,
        loop: PredictionLoop,
        logger: EventLogger | None = None,
        max_history_utterances: int = 20,
        history_char_budget: int = 1200,
        on_final=None,  # 発話確定時のフック(TTSキャッシュ無効化などに使う)
        lang: str = "ja",
    ):
        self._on_final = on_final
        self.lang = lang
        if lang == "en":
            from .text import en as _txt
        else:
            from .text import ja as _txt
        self._txt = _txt
        self._min_predict_chars = 6 if lang == "en" else 4
        self.asr = asr
        self.loop = loop
        self.log = logger or EventLogger(None)
        self.state = SessionState(started_t_ms=now_ms())
        self.tracker = StablePrefixTracker()
        self.max_history_utterances = max_history_utterances
        self.history_char_budget = history_char_budget
        self._last_asr_t: float | None = None
        self._tasks: list[asyncio.Task] = []

    # ---------- lifecycle ----------

    async def start(self) -> None:
        await self.asr.start()
        self._tasks = [
            asyncio.create_task(self.loop.run(), name="prediction-loop"),
            asyncio.create_task(self._consume_asr(), name="asr-consumer"),
        ]
        self.log.log("session_start")

    async def wait(self) -> None:
        """ASRイベント終了(typed/wav)まで待つ。micでは戻らない。"""
        await self._tasks[1]

    async def stop(self) -> None:
        self.log.log("session_stop")
        await self.asr.stop()
        self.loop.stop()
        for t in self._tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---------- internals ----------

    def reset(self) -> None:
        """履歴・認識状態のクリア(UIのリセット操作から)。"""
        self.state.history.clear()
        self.state.partial = ""
        self.state.stable = ""
        self.tracker.reset()
        self.log.log("session_reset")

    def history_text(self) -> str:
        """発話履歴を自然な文章に結合する。

        Azureのfinalは断片ごとに「。」を付けてくる(「の進捗状況を。」等)ため、
        末尾句読点を一旦剥がし、文として完結している断片の後にだけ「。」を打つ。
        未完結断片は次の断片とそのまま連結する(細切れ確定の緩和)。
        """
        parts: list[str] = []
        for utt in self.state.history:
            s = utt.rstrip("。、！？!?., ")
            if not s:
                continue
            if self.lang == "en":
                # 句読点は既に剥がしたので、元発話の終止符で完結を判定する
                # (en.is_incompleteは無終止符をほぼ常に未完結と見なすため使えない)
                had_terminal = utt.rstrip().endswith((".", "!", "?"))
                parts.append(s + ("." if had_terminal else ""))
                continue
            if parts:
                # 断片境界の助詞重複を縫合(「〜を」+「を報告します」→「〜を報告します」)
                s = ja.dedup_boundary_particle(parts[-1], s)
                if not s:
                    continue
            parts.append(s + ("。" if not ja.is_incomplete(s) else ""))
        text = (" ".join(parts) if self.lang == "en" else "".join(parts))
        if len(text) > self.history_char_budget:
            text = text[-self.history_char_budget:]
        return text

    async def _consume_asr(self) -> None:
        async for ev in self.asr.events():
            self._on_asr_event(ev)

    def _on_asr_event(self, ev: ASREvent) -> None:
        self.state.asr_events += 1
        if self._last_asr_t is not None:
            self.state.asr_intervals_ms.append(ev.t_ms - self._last_asr_t)
        self._last_asr_t = ev.t_ms

        if ev.is_final:
            self.log.log("asr_final", text=ev.text)
            if ev.text:
                self.state.history.append(ev.text)
                self.state.history = self.state.history[-self.max_history_utterances:]
                self.state.last_final_text = ev.text
                self.state.last_final_t_ms = ev.t_ms
            self.state.partial = ""
            self.state.stable = ""
            self.tracker.reset()
            if self._on_final:
                self._on_final(ev.text)
            return

        if not ev.text:
            return
        self.log.log("asr_partial", text=ev.text)
        if ev.text != self.state.partial:
            self.state.last_activity_t_ms = ev.t_ms
        self.state.partial = ev.text
        upd = self.tracker.update(ev.text, ev.t_ms)
        self.state.stable = upd.stable
        if upd.rewound:
            self.log.log("asr_rewind", stable=upd.stable)
        if upd.should_predict:
            # フィラー(えっと/あの…)に予測を条件付けさせない: 末尾フィラーを剥がした
            # テキストで予測する。表示はev.textのまま。全部フィラーなら送らない。
            predict_input = self._txt.strip_trailing_fillers(ev.text)
            if predict_input != ev.text:
                self.log.log("filler_stripped", raw=ev.text, input=predict_input)
            # 発話開始直後の1〜3文字(「新」「速」等)からの予測は語中カット誤りが
            # 多発するため見送る(履歴があっても最悪クラスの候補源になる)
            if len(self._txt.normalize_for_match(predict_input)) >= self._min_predict_chars:
                self._last_submit_input = predict_input
                self.loop.submit(self.history_text(), predict_input)

    def flush(self, t_ms: float, silence_ms: float = 350.0) -> None:
        """無音時に未送信の末尾を予測へ流す。

        「最後の1文字だけ伸びて詰まった」「フィラーで詰まった」場合、イベント駆動の
        発火条件は無音中に満たされないため、詰まりの最終形が予測されないまま残る。
        AppStateのtickから呼ばれ、その取り残しを解消する(自動発話の鮮度条件の前提)。
        """
        st = self.state
        if not st.partial or not st.last_activity_t_ms:
            return
        if t_ms - st.last_activity_t_ms < silence_ms:
            return
        predict_input = self._txt.strip_trailing_fillers(st.partial)
        if not predict_input or predict_input == getattr(self, "_last_submit_input", None):
            return
        if len(self._txt.normalize_for_match(predict_input)) < self._min_predict_chars:
            return
        self._last_submit_input = predict_input
        self.log.log("flush_submit", input=predict_input)
        self.loop.submit(self.history_text(), predict_input)
