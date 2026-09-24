"""Web UI サーバ(FastAPI + WebSocket)。

役割は「表示とゲート操作」のみ。音声処理・予測・再生はすべてPython側で動き、
ブラウザは状態のミラーとボタンイベントの送信のみを行う(遅延に関与しない)。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .clock import now_ms
from .gate import Gate, GateSignals
from .score import percentile

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class AppState:
    """run_app.py が組み立てた実行部品の束。"""

    def __init__(self, session, loop, prefetcher, playback, gate: Gate, logger,
                 predictor=None, lang: str = "ja", ui_state_path: str | None = None):
        self._ui_state_path = ui_state_path  # None=永続化なし(テスト等)
        self.lang = lang
        if lang == "en":
            from .text import en as _txt
        else:
            from .text import ja as _txt
        self._txt = _txt
        self.session = session
        self.loop = loop
        self.prefetcher = prefetcher
        self.playback = playback
        self.gate = gate
        self.log = logger
        self.predictor = predictor
        self.context_text = ""
        # 低確信候補は既定で「薄く表示」(隠さない)。隠すのはオプトイン。
        # 0.25閾値で隠すと有効候補の約半分が消えることをセッション実測で確認済み。
        self.hide_low_conf = False
        self.min_display_confidence = 0.25
        self.button_pressed = False
        self.tts_enabled = True
        self._partial_at_fire = ""
        self.press_queued = False  # タップはイベントとして積む(tickのサンプリング漏れ防止)
        self.pending_text: str | None = None
        self._pending_since = 0.0
        self._partial_at_pin = ""
        self.pending_timeout_ms = 5000.0
        # 自動発話モード: 候補準備済み+無音auto_silence_ms継続で自動再生。
        # 再起動でOFFに戻るとブラウザのトグル表示と食い違い「オンなのに発話しない」
        # 事故になるため、前回セッションの設定をファイルから引き継ぐ
        self.auto_mode = self._load_ui_state().get("auto_mode", False)
        self.auto_silence_ms = 800.0
        self.auto_cooldown_ms = 2500.0
        self.auto_final_grace_ms = 4000.0  # ASR確定後もこの間は「確定文の続き」を自動発話可
        self._last_auto_fire_t = 0.0
        self._last_auto_key: tuple[str, str] | None = None
        self._assist_texts: list[str] = []  # 補助が発話した内容(表示で区別するため)
        # マイクノイズゲート(幻単語対策 — 0で無効。閾値は実測: 発話RMS~1600/無音<150)
        self.mic_gate_rms = 350
        self.mic_gate_hangover_ms = 600.0
        self._gate_open_until = 0.0
        # マイク健全性の可視化: フレームが届いているか/音量が出ているかをUIとログに出す。
        # 「アプリは動いているのにマイクだけ死んでいる」事故を無言で起こさないため
        self.mic_frames = 0
        self.mic_level = 0            # 直近フレームのRMS
        self.mic_peak = 0             # 直近1秒のピークRMS
        self._mic_peak_since = 0.0
        self._last_mic_frame_t = 0.0
        self._mic_stall_logged = False
        self.mic = None               # run_appがMicSourceを差す(デバイス情報の参照用)
        # ボイス管理
        self.voices: list[str] = []
        self.active_voice: str = getattr(getattr(prefetcher, "tts", None), "voice", "") or ""
        self._rec: dict | None = None  # 録音中: {"name","until_ms","buf"}
        try:
            self._aio_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._aio_loop = None  # テスト等、ループ外での構築
        self.recording_text = (
            "Hello, this is a reference recording for voice synthesis. "
            "I usually speak at this speed and pitch. "
            "The weather is nice today, so I am thinking about going for a walk."
        ) if lang == "en" else (
            "こんにちは。これは音声合成のための参照音声です。"
            "私は普段このような速さと声の高さで話しています。"
            "今日は天気が良いので、散歩に出かけようと思っています。"
        )

    def _displayed_candidate(self) -> str | None:
        res = self.loop.latest_result
        if not (res and res.best and res.best.speakable and res.best.text):
            return None
        # 低確信の候補は表示しない(怪しい候補を見せない。タップ対象からも外れる)
        if self.hide_low_conf and res.confidence < self.min_display_confidence:
            return None
        return res.best.text

    def reset(self) -> None:
        """画面リセット: 履歴・認識状態・候補・キャッシュ・pendingを全クリア。"""
        self.playback.stop()
        self._clear_pending()
        self.prefetcher.invalidate()
        self.session.reset()
        self.loop.latest_result = None
        self.log.log("ui_reset")

    def set_context(self, text: str) -> None:
        """話題メモ(原稿・要点)を予測の文脈に注入する。"""
        self.context_text = text.strip()[:1500]
        if self.predictor is not None:
            self.predictor.preamble = self.context_text
        # ASRにも語彙ヒントとして渡す(Azureフレーズリスト。固有名詞・専門語の
        # 誤認識対策 — アクセント起因の文字起こし崩れに対する最安の一手)
        asr = getattr(self.session, "asr", None)
        if asr is not None and hasattr(asr, "set_phrases"):
            import re as _re

            lines = [ln.strip() for ln in self.context_text.splitlines()
                     if 0 < len(ln.strip()) <= 100]
            words = _re.findall(r"[A-Za-z][A-Za-z'\-]{3,}", self.context_text)
            seen: dict[str, None] = {}
            for p in lines + words:
                seen.setdefault(p, None)
            asr.set_phrases(list(seen))
        self.log.log("context_set", chars=len(self.context_text))

    def _load_ui_state(self) -> dict:
        if not self._ui_state_path:
            return {}
        import json as _json
        from pathlib import Path as _P

        try:
            return _json.loads(_P(self._ui_state_path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_ui_state(self) -> None:
        if not self._ui_state_path:
            return
        import json as _json
        from pathlib import Path as _P

        try:
            p = _P(self._ui_state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            state = {**self._load_ui_state(), "auto_mode": self.auto_mode}
            if self.active_voice:
                state["voice"] = self.active_voice  # 次回起動時の既定の声(run_appが参照)
            p.write_text(_json.dumps(state, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # UI設定の永続化失敗は動作に影響させない

    def _in_final_grace(self, st) -> bool:
        """ASR確定直後の自動発話猶予内か(候補の内容検証は_candidate_is_fresh側)。"""
        return bool(getattr(st, "last_final_text", "")
                    and now_ms() - st.last_final_t_ms <= self.auto_final_grace_ms)

    def _fire(self, cache, st) -> None:
        signals = GateSignals(button_pressed=True, cache_available=True,
                              playing=self.playback.is_playing)
        if self.gate.should_fire(signals):
            self._partial_at_fire = st.partial
            self.pending_text = None
            self.playback.play(cache.pcm_s16le, label=cache.text)
            self.log.log("assist_play", text=cache.text, cache_age_ms=round(cache.age_ms(), 1))

    def _clear_pending(self) -> None:
        self.pending_text = None
        self.prefetcher.clear_pin()

    # ---- マイク入力の一元処理(run_appのon_frameから毎フレーム呼ばれる) ----
    def feed_mic(self, pcm: bytes) -> None:
        asr = self.session.asr
        self._note_mic_frame(pcm)
        if self._rec is not None:
            self._rec["buf"] += pcm
            asr.push_audio(b"\x00" * len(pcm))  # 録音中は認識に流さない(履歴汚染防止)
            if now_ms() >= self._rec["until_ms"]:
                rec, self._rec = self._rec, None
                if self._aio_loop is not None:  # 音声スレッドからイベントループへ委譲
                    self._aio_loop.call_soon_threadsafe(
                        asyncio.ensure_future, self._finish_recording(rec))
        elif self.playback.is_playing:
            asr.push_audio(b"\x00" * len(pcm))  # 半二重(エコー再入力防止)
        else:
            asr.push_audio(self._noise_gate(pcm))

    def _note_mic_frame(self, pcm: bytes) -> None:
        """マイクフレームの到着と音量を記録する(UI表示・stall検出用)。"""
        import audioop

        rms = audioop.rms(pcm, 2)
        t = now_ms()
        self.mic_frames += 1
        self.mic_level = rms
        if t - self._mic_peak_since > 1000:
            self.mic_peak = rms
            self._mic_peak_since = t
        else:
            self.mic_peak = max(self.mic_peak, rms)
        self._last_mic_frame_t = t
        if self._mic_stall_logged:
            self._mic_stall_logged = False
            self.log.log("mic_resumed", frames=self.mic_frames)

    def mic_status(self) -> dict:
        """マイクの健全性(UIチップ用)。stalled=フレームが1秒以上届いていない。"""
        if not self._last_mic_frame_t:
            return {"frames": 0, "level": 0, "peak": 0, "stalled": True, "gate_rms": self.mic_gate_rms}
        return {
            "frames": self.mic_frames,
            "level": self.mic_level,
            "peak": self.mic_peak,
            "stalled": now_ms() - self._last_mic_frame_t > 1000,
            "gate_rms": self.mic_gate_rms,
        }

    def _noise_gate(self, pcm: bytes) -> bytes:
        """無音・呼吸音をASRに流さない(発話中のみ開くゲート)。

        ノイズからASRが幻の単語(Vosk英語の「the」等)を生成すると、partialが
        更新されて無音タイマーがリセットされ、自動発話が永遠に発火しなくなる。
        閾値超えで即開き、下回ってもhangoverの間は開いたまま(語尾の欠落防止)。
        """
        if self.mic_gate_rms <= 0:
            return pcm
        import audioop

        if audioop.rms(pcm, 2) >= self.mic_gate_rms:
            self._gate_open_until = now_ms() + self.mic_gate_hangover_ms
            return pcm
        if now_ms() < self._gate_open_until:
            return pcm
        return b"\x00" * len(pcm)

    # ---- ボイス管理 ----
    async def refresh_voices(self) -> None:
        tts = getattr(self.prefetcher, "tts", None)
        if tts is None or not hasattr(tts, "list_voices"):
            return
        try:
            self.voices = await tts.list_voices()
        except Exception as e:
            self.log.log("voices_error", error=str(e)[:200])

    def set_voice(self, voice: str) -> None:
        tts = getattr(self.prefetcher, "tts", None)
        if tts is None or not voice:
            return
        tts.voice = voice
        self.active_voice = voice
        # 旧声のキャッシュ・pendingは無効(声が変わる)
        self.prefetcher.invalidate()
        self._clear_pending()
        self._save_ui_state()  # 最後に選んだ声を次回起動時の既定にする
        self.log.log("voice_set", voice=voice)

    def start_voice_recording(self, name: str, seconds: float = 10.0) -> bool:
        name = name.strip()[:24]
        if not name or self._rec is not None or self.playback.is_playing:
            return False
        self._rec = {"name": name, "until_ms": now_ms() + seconds * 1000, "buf": b""}
        self.log.log("voice_record_start", name=name, seconds=seconds)
        return True

    async def _finish_recording(self, rec: dict) -> None:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(rec["buf"])
        tts = getattr(self.prefetcher, "tts", None)
        try:
            await tts.register_voice(buf.getvalue(), rec["name"])
            await self.refresh_voices()
            self.set_voice(rec["name"])
            self.log.log("voice_registered", name=rec["name"],
                         duration_s=round(len(rec["buf"]) / 32000, 1))
        except Exception as e:
            self.log.log("voice_register_error", name=rec["name"], error=str(e)[:200])

    def _spoken_overlap_remainder(self, cache, st) -> str | None:
        """候補の頭が「既に本人が発話した部分」と重なる場合、残り部分を返す。

        戻り値: None=重複なし / ""=全部発話済み / それ以外=残りテキスト
        """
        _ja = self._txt

        np_ = _ja.normalize_for_match(st.partial)
        ns = _ja.normalize_for_match(cache.source_input)
        if not ns or not np_.startswith(ns):
            return None
        delta = np_[len(ns):]
        if not delta:
            return None
        if len(delta) < 2 and delta not in _ja.PARTICLES:
            return None  # 助詞以外の1文字差分は誤削除リスクの方が大きい(助詞は「をを」防止で削る)
        nc = _ja.normalize_for_match(cache.text)
        if not nc.startswith(delta):
            return None
        # cache.textの生文字列上で、正規化deltaぶんを読み飛ばした位置を求める
        consumed = 0
        for i, ch in enumerate(cache.text):
            if consumed >= len(delta):
                return cache.text[i:].lstrip("、。・ ")
            if _ja.normalize_for_match(ch):
                consumed += 1
        return ""

    # ---- 20Hzで呼ばれる制御tick(発話判定・バージイン) ----
    # ボタンの意味は「いま表示されている候補を話す」。キャッシュが表示と一致すれば
    # 即再生、まだなら合成完了を待って再生(押している間)。表示と音声のズレを許さない。
    def _candidate_is_fresh(self, st) -> bool:
        """最新候補が現在のpartial全体を反映しているか(古い入力由来の候補を掴まない)。

        予測入力は末尾フィラー除去後のテキストなので、比較も同じ変換を通す。
        """
        res = self.loop.latest_result
        if not res:
            return False
        expected = self._txt.normalize_for_match(self._txt.strip_trailing_fillers(st.partial))
        if not expected:
            # ASR確定(ピリオド付与)でpartialが空になった直後の猶予: 候補が
            # 「いま確定した文の続き」なら鮮度ありとみなす。これが無いと
            # 自動発話の窓が「無音0.8s〜確定2.8s」の~2秒しかなく、予測+合成の
            # チェーンが間に合わないと確定と同時に永遠に発火不能になる。
            if (getattr(st, "last_final_text", "")
                    and now_ms() - st.last_final_t_ms <= self.auto_final_grace_ms):
                expected = self._txt.normalize_for_match(
                    self._txt.strip_trailing_fillers(st.last_final_text))
            if not expected:
                return False
        got = self._txt.normalize_for_match(res.input_latest)
        if got == expected:
            return True
        # 回線遅延への許容: 候補入力が現partialの接頭辞で未反映の末尾が数文字なら
        # 実質最新とみなす(遠隔では予測往復>1sとなりcandidate_staleが自動発話を
        # 全滅させる)。冒頭重複は再生前の_spoken_overlap_remainderが除去する。
        tol = 3 if self.lang == "en" else 2
        return bool(got) and expected.startswith(got) and len(expected) - len(got) <= tol

    def tick(self) -> None:
        st = self.session.state
        disp = self._displayed_candidate()
        cur = self.prefetcher.current

        # マイクのフレームが途絶えたら一度だけ記録する(デバイス切替・ストリーム停止の検知)
        if self._last_mic_frame_t and not self._mic_stall_logged \
                and now_ms() - self._last_mic_frame_t > 3000:
            self._mic_stall_logged = True
            self.log.log("mic_stalled", last_frame_ago_ms=round(now_ms() - self._last_mic_frame_t),
                         frames=self.mic_frames)

        # 無音時に未予測の末尾を予測へ流す(詰まりの最終形を候補に反映させる前提処理)
        if hasattr(self.session, "flush"):
            self.session.flush(now_ms())
        self.playback.pop_completed()  # 完了フラグは消費のみ(履歴コミットは廃止)

        # タップ方式: 押した瞬間の表示候補を「発話するコミット」として扱う。
        # 離しても再生は続く/合成完了を待って自動再生。もう一度押すとキャンセル。
        press_edge = self.press_queued
        self.press_queued = False

        if press_edge:
            if self.playback.is_playing:
                self.playback.stop()
                self._clear_pending()
                self.log.log("assist_stop", reason="user_cancel")
            elif self.pending_text:
                self._clear_pending()
                self.log.log("assist_cancel_pending")
            elif disp:
                self.pending_text = disp
                self._pending_since = now_ms()
                self._partial_at_pin = st.partial
                if not (cur and cur.text == disp):
                    self.prefetcher.request_pin(disp, st.partial)
                    self.log.log("assist_pending", text=disp)
            else:
                cache = self.prefetcher.fresh(st.partial)
                if cache:
                    self._fire(cache, st)

        # 自動発話: 発話が止まったら表示中の候補を話す。
        # 音声が準備済みなら即再生、まだなら自動ピン(合成完了次第、pending解決パスが再生)。
        # 音声準備の完了(~1.5s)を待たず詰まりの瞬間に意思決定するのが要点 —
        # 待っているとASRの文確定でpartialごと消える。確定後もfinal猶予の間は対象
        # (candidate鮮度側で「確定文の続き」であることを検証する)。
        if self.auto_mode and not self.playback.is_playing and not self.pending_text \
                and (st.partial or self._in_final_grace(st)):
            silence = now_ms() - st.last_activity_t_ms if st.last_activity_t_ms else 0.0
            # 再発話抑止キーは内容ベース(候補+予測入力)にする。partialを含めると
            # ASR確定で「本文→空」に変わった瞬間に別キー扱いとなり、確定前に話した
            # 続きをfinal猶予でもう一度話してしまう(ピリオド前後の二重発話)。
            res_now = self.loop.latest_result
            inp_norm = self._txt.normalize_for_match(res_now.input_latest) if res_now else ""
            key = (disp or "", inp_norm)
            blocked = None
            if silence < self.auto_silence_ms:
                blocked = "waiting_silence"
            elif not disp:
                blocked = "no_candidate"
            elif now_ms() - self._last_auto_fire_t < self.auto_cooldown_ms:
                blocked = "cooldown"
            elif key == self._last_auto_key:
                blocked = "same_key"
            elif not self._candidate_is_fresh(st):
                blocked = "candidate_stale"
            # 発火できない理由を~1秒おきに記録(無音が続いているのに鳴らない場合の自己診断)
            if blocked and blocked != "waiting_silence" \
                    and now_ms() - getattr(self, "_last_block_log_t", 0.0) > 1000:
                self._last_block_log_t = now_ms()
                self.log.log("auto_blocked", reason=blocked, silence_ms=round(silence))
            if blocked is None:
                self._last_auto_key = key
                self._last_auto_fire_t = now_ms()
                if not st.partial:
                    st.last_final_t_ms = 0.0  # final猶予での補助は1確定につき1回まで
                if cur and cur.text == disp and self._spoken_overlap_remainder(cur, st) is None:
                    self.log.log("assist_auto_fire", text=disp, silence_ms=round(silence))
                    self._fire(cur, st)
                else:
                    self.pending_text = disp
                    self._pending_since = now_ms()
                    self._partial_at_pin = st.partial
                    self.prefetcher.request_pin(disp, st.partial)
                    self.log.log("assist_auto_pin", text=disp, silence_ms=round(silence))

        # pending解決: ピンした候補の合成が完成したら再生(ボタン状態に依らない)
        if self.pending_text and not self.playback.is_playing:
            if cur and cur.text == self.pending_text:
                remainder = self._spoken_overlap_remainder(cur, st)
                if remainder is None:
                    self.prefetcher.clear_pin()
                    self._fire(cur, st)
                elif len(remainder) >= 3:
                    # 候補の冒頭を本人が既に話している → 残りだけ合成し直す
                    self.log.log("junction_overlap", cached=cur.text, remainder=remainder)
                    self.pending_text = remainder
                    self._pending_since = now_ms()
                    self._partial_at_pin = st.partial
                    self.prefetcher.request_pin(remainder, st.partial)
                else:
                    self.log.log("junction_overlap_all_spoken", cached=cur.text)
                    self._clear_pending()
            elif now_ms() - self._pending_since > self.pending_timeout_ms:
                self._clear_pending()
                self.log.log("assist_pending_timeout")

        # ユーザーの発話が明確に進んだら、pendingも再生も破棄(本人優先)。
        # (文字数同等の書き換えはAzureのpartial refinementなので無視)
        if self.pending_text and len(st.partial) > len(self._partial_at_pin) + 1:
            self._clear_pending()
            self.log.log("assist_cancel_pending", reason="user_resumed")
        if self.playback.is_playing and len(st.partial) > len(self._partial_at_fire) + 1:
            self.playback.stop()
            self.log.log("assist_stop", reason="user_resumed")

    def snapshot(self) -> dict:
        st = self.session.state
        res = self.loop.latest_result
        cache = self.prefetcher.current
        fresh = self.loop.stats.freshness_ms[-50:]
        walls = self.loop.stats.wall_ms[-50:]
        iv = st.asr_intervals_ms[-50:]
        asr_error = getattr(self.session.asr, "last_error", None)
        return {
            "history_text": self.session.history_text()[-160:],
            "partial": st.partial,
            "asr_error": asr_error,
            "stable": st.stable,
            "candidate": {
                "text": self._displayed_candidate() or "",
                "confidence": res.confidence if res else 0.0,
                "agree": res.agree_len if res else 0,
                "low_conf": bool(res) and res.confidence < self.min_display_confidence,
                "for_input": res.input_latest if res else "",
            },
            "stats": {
                "freshness_p50": round(percentile(fresh, 50)) if fresh else None,
                "wall_p50": round(percentile(walls, 50)) if walls else None,
                "asr_interval_p50": round(percentile(iv, 50)) if iv else None,
                "cycles": self.loop.stats.cycles,
                "errors": self.loop.stats.errors,
            },
            "tts": {
                "enabled": self.tts_enabled,
                "ready": cache is not None,
                "fresh": self.prefetcher.fresh(st.partial) is not None,
                "text": cache.text if cache else "",
                "age_ms": round(cache.age_ms()) if cache else None,
                "ttfa_ms": round(cache.ttfa_ms) if cache else None,
                "duration_ms": round(cache.duration_ms) if cache else None,
                "synth_count": self.prefetcher.synth_count,
                "error": self.prefetcher.last_error,
            },
            "playing": self.playback.is_playing,
            "playing_text": self.playback.playing_text,
            # 「今から/今」話す内容(合成待ち or 再生中)。UIはこれをゴーストに固定表示する
            "speak_text": self.playback.playing_text if self.playback.is_playing else self.pending_text,
            "button": self.button_pressed,
            "pending": self.pending_text,
            "cache_matches": bool(
                cand_text := self._displayed_candidate()
            ) and cache is not None and cache.text == cand_text,
            "gate_fired": self.gate.state.fired_count,
            "context_len": len(self.context_text),
            "hide_low_conf": self.hide_low_conf,
            "auto_mode": self.auto_mode,
            "silence_ms": round(now_ms() - st.last_activity_t_ms) if st.last_activity_t_ms else None,
            "voices": self.voices,
            "active_voice": self.active_voice,
            "mic": self.mic_status(),
            "recording": {
                "name": self._rec["name"],
                "remaining_s": max(0, round((self._rec["until_ms"] - now_ms()) / 1000, 1)),
                "text": self.recording_text,
            } if self._rec else None,
            "t_ms": round(now_ms()),
        }


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="PredictiveSpeaking")

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/shutdown")
    async def shutdown() -> dict:
        """新インスタンスからの引き継ぎ要求(旧プロセス残留によるbind失敗の根治)。"""
        cb = getattr(app.state, "shutdown_cb", None)
        if cb:
            cb()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()

        async def sender() -> None:
            while True:
                await websocket.send_text(json.dumps(state.snapshot(), ensure_ascii=False))
                await asyncio.sleep(0.1)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                if msg.get("type") == "button":
                    state.button_pressed = bool(msg.get("pressed"))
                    if state.button_pressed:
                        state.press_queued = True  # 短いタップでも必ず1押下として処理
                    state.log.log("ui_button", pressed=state.button_pressed)
                elif msg.get("type") == "tts_enabled":
                    state.tts_enabled = bool(msg.get("value"))
                    state.prefetcher.enabled = state.tts_enabled
                elif msg.get("type") == "reset":
                    state.reset()
                elif msg.get("type") == "context":
                    state.set_context(str(msg.get("text", "")))
                elif msg.get("type") == "hide_low_conf":
                    state.hide_low_conf = bool(msg.get("value"))
                elif msg.get("type") == "auto_mode":
                    state.auto_mode = bool(msg.get("value"))
                    state._save_ui_state()  # 再起動後も設定を引き継ぐ
                    state.log.log("ui_auto_mode", value=state.auto_mode)
                elif msg.get("type") == "set_voice":
                    state.set_voice(str(msg.get("voice", "")))
                elif msg.get("type") == "record_voice":
                    state.start_voice_recording(str(msg.get("name", "")),
                                                float(msg.get("seconds", 10)))
                elif msg.get("type") == "refresh_voices":
                    await state.refresh_voices()
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            state.button_pressed = False

    return app
