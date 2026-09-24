"""Azure Speech ストリーミングASR (ja-JP)。

PushAudioInputStreamに音声を流し、recognizing(partial)/recognized(final)を
SDKのコールバックスレッドからasyncioへ橋渡しする。
環境変数: AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (.env対応)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from ..clock import now_ms
from .base import ASREvent


class AzureSpeechASR:
    def __init__(self, sample_rate: int = 16000, language: str = "ja-JP",
                 segmentation_silence_ms: int = 2800, phrases: list[str] | None = None):
        self.sample_rate = sample_rate
        self.language = language
        # 既定(~500ms)だと言い淀みのたびに文が確定して継続点が壊れるため長めにする。
        # 詰まり(=無音)の間もpartialが生き続けることが介入の前提になる。
        self.segmentation_silence_ms = segmentation_silence_ms
        self.phrases: list[str] = list(phrases or [])
        self._events: asyncio.Queue[ASREvent] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recognizer = None
        self._push_stream = None
        self._phrase_grammar = None
        self.last_error: str | None = None

    def set_phrases(self, phrases: list[str]) -> None:
        """認識器に語彙ヒントを与える(話題メモ由来。アクセント・固有名詞の誤認識対策)。

        start()前なら保存のみ、稼働中ならその場でPhraseListGrammarを差し替える。
        """
        self.phrases = [p.strip() for p in phrases if p and p.strip()][:500]
        if self._recognizer is not None:
            self._apply_phrases()

    def _apply_phrases(self) -> None:
        import azure.cognitiveservices.speech as speechsdk

        if self._phrase_grammar is None:
            self._phrase_grammar = speechsdk.PhraseListGrammar.from_recognizer(self._recognizer)
        self._phrase_grammar.clear()
        for p in self.phrases:
            self._phrase_grammar.addPhrase(p)

    async def start(self) -> None:
        import azure.cognitiveservices.speech as speechsdk

        key = os.environ.get("AZURE_SPEECH_KEY")
        region = os.environ.get("AZURE_SPEECH_REGION", "japaneast")
        if not key:
            raise RuntimeError("AZURE_SPEECH_KEY が未設定です(.envを確認)")

        self._loop = asyncio.get_running_loop()
        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
        cfg.speech_recognition_language = self.language
        cfg.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(self.segmentation_silence_ms),
        )
        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate, bits_per_sample=16, channels=1)
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_cfg = speechsdk.audio.AudioConfig(stream=self._push_stream)
        self._recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

        def _emit(text: str, final: bool) -> None:
            if self._loop and text:
                self._loop.call_soon_threadsafe(
                    self._events.put_nowait,
                    ASREvent(text=text, is_final=final, t_ms=now_ms()))

        self._recognizer.recognizing.connect(lambda evt: _emit(evt.result.text, False))
        self._recognizer.recognized.connect(lambda evt: _emit(evt.result.text, True))

        def _on_canceled(evt) -> None:
            # 認証エラー等はここに来る。気づけるようにerrorへ記録する(UI側で表示)。
            self.last_error = f"canceled: {evt.reason} {getattr(evt, 'error_details', '')}"

        self._recognizer.canceled.connect(_on_canceled)
        if self.phrases:
            self._apply_phrases()
        self._recognizer.start_continuous_recognition_async()

    def push_audio(self, pcm_s16le: bytes) -> None:
        if self._push_stream:
            self._push_stream.write(pcm_s16le)

    async def events(self) -> AsyncIterator[ASREvent]:
        while True:
            yield await self._events.get()

    async def stop(self) -> None:
        if self._recognizer:
            self._recognizer.stop_continuous_recognition_async()
        if self._push_stream:
            self._push_stream.close()
