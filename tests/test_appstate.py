"""AppState.tick の発話フロー(即時再生 / 合成待ちpending / 解放停止)のテスト。"""

from __future__ import annotations

from types import SimpleNamespace

from predictive_speaking.clock import now_ms
from predictive_speaking.events import EventLogger
from predictive_speaking.gate import Gate
from predictive_speaking.tts.cache import CachedAudio
from predictive_speaking.webapp import AppState


class FakePlayback:
    def __init__(self):
        self.played: list[str] = []
        self._playing = False
        self.playing_text = None

    def play(self, pcm, label=None):
        self.played.append(label)
        self._playing = True
        self.playing_text = label

    def stop(self):
        self._playing = False
        self.playing_text = None
        self._completed = None

    def finish_naturally(self):
        self._completed = self.playing_text
        self._playing = False
        self.playing_text = None

    def pop_completed(self):
        c = getattr(self, "_completed", None)
        self._completed = None
        return c

    @property
    def is_playing(self):
        return self._playing


class FakePrefetcher:
    def __init__(self):
        self.cache: CachedAudio | None = None
        self.current = None
        self.synth_count = 0
        self.last_error = None
        self.enabled = True

    def set_cache(self, text: str, source_input: str):
        self.cache = CachedAudio(text=text, pcm_s16le=b"\x00" * 100, sample_rate=24000,
                                 source_input=source_input, created_t_ms=now_ms(),
                                 ttfa_ms=1, synth_ms=1)
        self.current = self.cache

    def fresh(self, partial, ttl_ms=6000.0):
        if self.cache and partial:
            return self.cache
        return None

    def request_pin(self, text, source_input):
        self.pinned = (text, source_input)

    def clear_pin(self):
        self.pinned = None

    def invalidate(self):
        self.cache = None
        self.current = None


def make_state(partial: str, candidate: str | None, confidence: float = 0.8,
               predictor=None):
    class AsrSink:
        def __init__(self):
            self.pushed: list[bytes] = []

        def push_audio(self, pcm):
            self.pushed.append(pcm)

    session = SimpleNamespace(
        state=SimpleNamespace(partial=partial, stable="", history=[],
                              asr_intervals_ms=[], asr_events=0,
                              last_activity_t_ms=now_ms()),
        history_text=lambda: "",
        asr=AsrSink(),
    )
    session.reset = lambda: setattr(session, "was_reset", True)
    session.flush = lambda t_ms, silence_ms=350.0: None
    best = SimpleNamespace(text=candidate, speakable=True) if candidate else None
    loop = SimpleNamespace(
        latest_result=SimpleNamespace(best=best, input_latest=partial,
                                      confidence=confidence, agree_len=3) if best else None,
        stats=SimpleNamespace(freshness_ms=[], wall_ms=[], cycles=0,
                              errors=0, stale_publishes=0),
    )
    return AppState(session, loop, FakePrefetcher(), FakePlayback(), Gate(cooldown_ms=0),
                    EventLogger(None), predictor=predictor)


def test_instant_play_when_cache_matches_display():
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.played == ["すね散歩に行きましょう"]


def test_release_does_not_stop_playback():
    """タップ方式: 離しても再生は最後まで続く。"""
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    app.button_pressed = False
    app.tick()
    assert app.playback.is_playing  # 離しても止まらない


def test_pending_survives_release():
    """すぐ離しても、コミットした候補は合成完了後に再生される。"""
    app = make_state("今日はいい天気で", "新しい候補テキスト")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.pending_text == "新しい候補テキスト"
    app.button_pressed = False
    app.tick()
    assert app.pending_text == "新しい候補テキスト"  # 離してもpending維持
    app.prefetcher.set_cache("新しい候補テキスト", "今日はいい天気で")
    app.tick()
    assert app.playback.played == ["新しい候補テキスト"]


def test_second_press_cancels():
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.is_playing
    app.button_pressed = False
    app.tick()
    app.button_pressed = True; app.press_queued = True  # 再押下=停止
    app.tick()
    assert not app.playback.is_playing


def test_pending_pins_press_moment_candidate():
    app = make_state("今日はいい天気で", "新しい候補テキスト")
    app.prefetcher.set_cache("古い候補", "今日はいい")  # 表示と不一致
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.played == []          # 古いキャッシュは再生しない
    assert app.pending_text == "新しい候補テキスト"
    assert app.prefetcher.pinned[0] == "新しい候補テキスト"  # ピン留め合成を要求
    # 表示が先に動いてもピンは変わらない(押した瞬間の候補を話す契約)
    app.loop.latest_result.best.text = "さらに新しい候補"
    app.tick()
    assert app.pending_text == "新しい候補テキスト"
    app.prefetcher.set_cache("新しい候補テキスト", "今日はいい天気で")  # ピンの合成完了
    app.tick()
    assert app.playback.played == ["新しい候補テキスト"]
    assert app.pending_text is None


def test_junction_overlap_resynthesizes_remainder():
    """候補の冒頭を本人が既に話していたら、残りだけを合成し直して話す。"""
    app = make_state("今日はいい天気ですが明日は", "天気ですが明日は雨でしょう")
    app.prefetcher.set_cache("天気ですが明日は雨でしょう", "今日はいい")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.played == []            # 重複したまま再生しない
    assert app.pending_text == "雨でしょう"      # 残り部分を再ピン
    assert app.prefetcher.pinned[0] == "雨でしょう"
    app.prefetcher.set_cache("雨でしょう", "今日はいい天気ですが明日は")
    app.tick()
    assert app.playback.played == ["雨でしょう"]


def test_single_particle_overlap_strips_at_play():
    """候補生成後にユーザーが助詞を1つ言い足した場合、重複助詞を除いて残りを再ピン。"""
    app = make_state("本日の進捗状況を", "を報告いたします")
    app.prefetcher.set_cache("を報告いたします", "本日の進捗状況")  # 「を」の前が由来
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.played == []
    assert app.pending_text == "報告いたします"  # 重複「を」を除いた残り
    app.prefetcher.set_cache("報告いたします", "本日の進捗状況を")
    app.tick()
    assert app.playback.played == ["報告いたします"]


def test_auto_mode_fires_on_silence():
    """自動モード: 候補準備済み+無音継続でボタンなしで発話する。"""
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000  # 1秒無音
    app.tick()
    assert app.playback.played == ["すね散歩に行きましょう"]


def test_auto_mode_waits_while_speaking():
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms()  # 直前まで発話中
    app.tick()
    assert app.playback.played == []


def test_auto_mode_no_repeat_same_candidate():
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.auto_mode = True
    app.auto_cooldown_ms = 0
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.tick()
    assert len(app.playback.played) == 1
    app.playback.stop()  # 再生終了後も同じ候補は繰り返さない
    app.tick()
    assert len(app.playback.played) == 1


def test_auto_mode_pins_when_audio_not_ready():
    """自動モード: 音声未準備なら自動ピンし、合成完了次第再生する。"""
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("古い候補", "昔の入力")  # 表示と不一致
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.tick()
    assert app.playback.played == []
    assert app.pending_text == "すね散歩に行きましょう"       # 自動ピン
    assert app.prefetcher.pinned[0] == "すね散歩に行きましょう"
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.tick()
    assert app.playback.played == ["すね散歩に行きましょう"]  # 完成次第自動再生


def test_auto_waits_for_fresh_candidate():
    """候補が現在のpartialを反映するまで自動発火しない(古い候補を掴まない)。"""
    app = make_state("今日はいい天気ですが明日は", "すね散歩に行きましょう")
    app.prefetcher.set_cache("古い別候補", "今日はいい")  # 音声は未準備(不一致)
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "今日はいい天気で"  # 古い入力由来
    app.tick()
    assert app.playback.played == [] and app.pending_text is None
    app.loop.latest_result.input_latest = "今日はいい天気ですが明日は"  # 新鮮になった
    app.tick()
    assert app.pending_text == "すね散歩に行きましょう"  # 自動ピン発動


def test_auto_tolerates_slightly_stale_candidate():
    """回線遅延許容: 候補入力が現partialの接頭辞で差分2文字以内なら発火する。"""
    app = make_state("今日はいい天気です", "ね散歩に行きましょう")
    app.prefetcher.set_cache("古い別候補", "昔の入力")  # 音声は未準備
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "今日はいい天気で"  # 差分「す」=1文字
    app.tick()
    assert app.pending_text == "ね散歩に行きましょう"  # stale扱いせず自動ピン


def test_auto_still_blocks_clearly_stale_candidate():
    """差分が許容幅(ja:2文字)を超える古い候補は従来通りブロックする。"""
    app = make_state("今日はいい天気ですがね", "候補テキストです")
    app.prefetcher.set_cache("古い別候補", "昔の入力")
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "今日はいい天気で"  # 差分「すがね」=3文字
    app.tick()
    assert app.playback.played == [] and app.pending_text is None


def test_auto_blocks_non_prefix_stale_candidate():
    """差分が小さくても接頭辞関係にない(巻き戻り等)候補はブロックする。"""
    app = make_state("今日はいい天気だ", "候補テキストです")
    app.prefetcher.set_cache("古い別候補", "昔の入力")
    app.auto_mode = True
    app.session.state.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "今日はいい天気で"  # 長さ近いが接頭辞でない
    app.tick()
    assert app.playback.played == [] and app.pending_text is None


def test_auto_fires_in_post_final_grace():
    """ASR確定でpartialが空になっても、候補が確定文の続きなら猶予内は自動発話する。"""
    app = make_state("", "and the next step is testing")
    app.prefetcher.set_cache("古い候補", "昔の入力")  # 音声未準備→自動ピン経路
    app.auto_mode = True
    st = app.session.state
    st.last_final_text = "The quality of this item is good."
    st.last_final_t_ms = now_ms() - 1000  # 猶予4秒以内
    st.last_activity_t_ms = now_ms() - 1000  # 無音0.8秒は経過
    app.loop.latest_result.input_latest = "The quality of this item is good"
    app.tick()
    assert app.pending_text == "and the next step is testing"  # 自動ピン発動


def test_auto_does_not_fire_after_grace_expires():
    app = make_state("", "and the next step is testing")
    app.prefetcher.set_cache("古い候補", "昔の入力")
    app.auto_mode = True
    st = app.session.state
    st.last_final_text = "The quality of this item is good."
    st.last_final_t_ms = now_ms() - 10_000  # 猶予切れ
    st.last_activity_t_ms = now_ms() - 10_000
    app.loop.latest_result.input_latest = "The quality of this item is good"
    app.tick()
    assert app.playback.played == [] and app.pending_text is None


def test_auto_grace_requires_candidate_from_final():
    """確定文と無関係な入力から出た候補は猶予内でも発話しない。"""
    app = make_state("", "totally unrelated continuation")
    app.prefetcher.set_cache("古い候補", "昔の入力")
    app.auto_mode = True
    st = app.session.state
    st.last_final_text = "The quality of this item is good."
    st.last_final_t_ms = now_ms() - 1000
    st.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "something completely different"
    app.tick()
    assert app.playback.played == [] and app.pending_text is None


def test_auto_no_double_fire_across_final():
    """確定前に話した続きを、final猶予でもう一度話さない(ピリオド前後の二重発話防止)。"""
    app = make_state("the quality of this item is good", "and the delivery was quick")
    app.prefetcher.set_cache("and the delivery was quick", "the quality of this item is good")
    app.auto_mode = True
    app.auto_cooldown_ms = 0  # クールダウンでは防がれない状況を作る
    st = app.session.state
    st.last_activity_t_ms = now_ms() - 1000
    app.tick()
    assert len(app.playback.played) == 1  # 確定前に1回発話
    app.playback.stop()
    # ASR確定をシミュレート: partial消滅+final記録(入力・候補は変わらない)
    st.partial = ""
    st.last_final_text = "The quality of this item is good."
    st.last_final_t_ms = now_ms() - 100
    app.tick()
    assert len(app.playback.played) == 1  # 同じ続きは再発話しない(same_key)


def test_auto_grace_assists_once_per_final():
    """final猶予内の自動補助は1確定につき1回で打ち止め。"""
    app = make_state("", "and the next step is testing")
    app.prefetcher.set_cache("and the next step is testing",
                             "the quality of this item is good")
    app.auto_mode = True
    app.auto_cooldown_ms = 0
    st = app.session.state
    st.last_final_text = "The quality of this item is good."
    st.last_final_t_ms = now_ms() - 500
    st.last_activity_t_ms = now_ms() - 1000
    app.loop.latest_result.input_latest = "The quality of this item is good"
    app.tick()
    assert len(app.playback.played) == 1
    assert st.last_final_t_ms == 0.0  # 猶予は閉じられた
    app.playback.stop()
    # 別の候補に差し替わっても、この確定に対する再補助はしない
    app.loop.latest_result.best.text = "so let me summarize the results"
    app.prefetcher.set_cache("so let me summarize the results",
                             "the quality of this item is good")
    app.tick()
    assert len(app.playback.played) == 1


def test_mic_status_tracks_frames_and_stall():
    """マイクの健全性がsnapshotに出る(未接続→受信→停止)。"""
    import array

    app = make_state("hello", None)
    assert app.mic_status()["stalled"] is True and app.mic_status()["frames"] == 0
    loud = array.array("h", [3000] * 320).tobytes()
    app.feed_mic(loud)
    st = app.mic_status()
    assert st["frames"] == 1 and st["stalled"] is False and st["peak"] >= 2900
    app._last_mic_frame_t = now_ms() - 5000  # フレーム途絶
    assert app.mic_status()["stalled"] is True
    app.tick()
    assert app._mic_stall_logged is True  # 一度だけ記録される


def test_noise_gate_blocks_quiet_frames():
    """無音・呼吸音レベルのフレームはASRへゼロ埋めで渡す(幻単語対策)。"""
    import array

    app = make_state("hello", None)
    app._gate_open_until = 0.0  # hangoverなしの状態から
    quiet = array.array("h", [50] * 320).tobytes()   # RMS=50 < 350
    loud = array.array("h", [3000] * 320).tobytes()  # RMS=3000 >= 350
    app.feed_mic(quiet)
    assert app.session.asr.pushed[-1] == b"\x00" * len(quiet)
    app.feed_mic(loud)
    assert app.session.asr.pushed[-1] == loud  # 発話で即開く
    app.feed_mic(quiet)
    assert app.session.asr.pushed[-1] == quiet  # hangover中は開いたまま(語尾保護)
    app._gate_open_until = now_ms() - 1  # hangover切れ
    app.feed_mic(quiet)
    assert app.session.asr.pushed[-1] == b"\x00" * len(quiet)


def test_noise_gate_disabled_passes_everything():
    import array

    app = make_state("hello", None)
    app.mic_gate_rms = 0
    quiet = array.array("h", [10] * 320).tobytes()
    app.feed_mic(quiet)
    assert app.session.asr.pushed[-1] == quiet


def test_manual_mode_does_not_auto_fire():
    app = make_state("今日はいい天気で", "すね散歩に行きましょう")
    app.prefetcher.set_cache("すね散歩に行きましょう", "今日はいい天気で")
    app.session.state.last_activity_t_ms = now_ms() - 5000
    app.tick()
    assert app.playback.played == []  # auto_mode=False(既定)では自発しない


def test_feed_mic_routing():
    """通常=素通し / 再生中=無音 / 録音中=バッファ+無音。"""
    app = make_state("発話中", "候補のテキスト")
    frame = b"\x01\x02" * 160
    app.feed_mic(frame)
    assert app.session.asr.pushed[-1] == frame  # 素通し
    app.playback.play(b"\x00" * 100, label="x")
    app.feed_mic(frame)
    assert app.session.asr.pushed[-1] == b"\x00" * len(frame)  # 半二重
    app.playback.stop()
    assert app.start_voice_recording("guest", seconds=5)
    app.feed_mic(frame)
    assert app._rec["buf"] == frame                      # 録音バッファ
    assert app.session.asr.pushed[-1] == b"\x00" * len(frame)  # 認識には流さない


def test_set_voice_invalidates_cache():
    app = make_state("発話中", "候補のテキスト")
    app.prefetcher.tts = SimpleNamespace(voice="me")
    app.prefetcher.set_cache("候補のテキスト", "発話中")
    app.set_voice("guest")
    assert app.prefetcher.tts.voice == "guest"
    assert app.active_voice == "guest"
    assert app.prefetcher.current is None  # 旧声のキャッシュは破棄


def test_recording_guards():
    app = make_state("発話中", "候補")
    assert not app.start_voice_recording("")           # 空名は不可
    app.playback.play(b"\x00" * 10, label="x")
    assert not app.start_voice_recording("guest")      # 再生中は不可


def test_barge_in_on_partial_growth():
    app = make_state("今日はいい天気で", "候補のテキスト")
    app.prefetcher.set_cache("候補のテキスト", "今日はいい天気で")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.is_playing
    app.session.state.partial = "今日はいい天気ですが明日"  # 発話が明確に進む
    app.tick()
    assert not app.playback.is_playing


def test_reset_clears_everything():
    app = make_state("発話中のテキスト", "候補のテキスト")
    app.prefetcher.set_cache("候補のテキスト", "発話中のテキスト")
    app.button_pressed = True; app.press_queued = True
    app.tick()
    assert app.playback.is_playing
    app.reset()
    assert not app.playback.is_playing
    assert app.prefetcher.current is None
    assert app.pending_text is None
    assert app.loop.latest_result is None
    assert getattr(app.session, "was_reset", False)


def test_low_confidence_candidate_faint_by_default():
    app = make_state("入力テキスト", "怪しい候補", confidence=0.1)
    assert app._displayed_candidate() == "怪しい候補"  # 既定: 薄く表示(隠さない)
    app.hide_low_conf = True
    assert app._displayed_candidate() is None       # オプトインで隠す


def test_set_context_updates_predictor_preamble():
    pred = SimpleNamespace(preamble="")
    app = make_state("入力", "候補", predictor=pred)
    app.set_context("  今日の発表の要点メモ  ")
    assert pred.preamble == "今日の発表の要点メモ"
    assert app.context_text == "今日の発表の要点メモ"
