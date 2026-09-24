"""llama-server バックエンド。

KV常駐の実体は llama-server の cache_prompt + slot固定:
- 各候補ストリーム(greedy / sample_i)を同じ id_slot に送り続けると、
  サーバ側で前回プロンプトとの最長共通prefixのKVが再利用され、
  毎サイクルのプリフィルは「新しく増えた末尾トークンのみ」になる。
- 巻き戻り(ASR書き換え)も共通prefix計算で自動的に吸収される。

サーバ起動要件: llama-server --parallel <n_candidates以上> (例: -np 3)
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from ..clock import now_ms
from ..validate import agreement_len, confidence, validate
from .types import Candidate, CandidateSet

# 文末記号では止めない: 文がほぼ完結した位置では「語尾の残り」しか得られないため、
# 次の文の内容まで生成し、どこを候補にするかはvalidator側で選ぶ。
DEFAULT_STOP = ["\n"]

CHAT_SYSTEM_PROMPT = (
    "あなたは日本語の発話継続予測エンジンです。対話相手ではありません。\n"
    "入力は同じ話者が現在話している途中の書き起こしです。\n"
    "同じ話者がこの直後に言い足す短い続き(1〜3文節、最大12文字程度)だけを出力してください。\n"
    "返答・相槌・質問・説明・引用符は禁止。すでに話した部分を繰り返さない。\n"
    "続きが一意に定まらない場合や文が完結している場合は <SILENCE> とだけ出力する。\n"
    "/no_think"
)


class LlamaCppPredictor:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        mode: str = "raw",              # "raw"(base/CPT向け素の継続) | "chat"(instruct向け)
        n_candidates: int = 3,
        n_predict: int = 24,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_chars: int | None = None,   # None=言語別既定(ja:22 / en:60)
        preamble: str = "",             # raw modeでprompt先頭に置く文脈ヒント(空=純粋な継続)
        timeout_s: float = 30.0,
        lang: str = "ja",
        repeat_penalty: float = 1.0,    # 日本語は助詞・かなの正常な繰り返しが多く、罰すると不自然化する
        dry_multiplier: float = 0.75,   # 反復退化はDRYサンプラーで抑止(正常な繰り返しは罰しない)
        n_probs: int = 1,               # 0にするとlogprob信頼度を諦める(投機的デコード互換性検証用)
    ):
        self.repeat_penalty = repeat_penalty
        self.dry_multiplier = dry_multiplier
        self.n_probs = n_probs
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.n_candidates = max(1, n_candidates)
        self.n_predict = n_predict
        self.temperature = temperature
        self.top_p = top_p
        self.lang = lang
        self.max_chars = max_chars if max_chars is not None else (60 if lang == "en" else 22)
        self.preamble = preamble
        self._client = httpx.AsyncClient(timeout=timeout_s)

    # ---------- public ----------

    async def predict(self, history: str, latest_partial: str, request_id: int) -> CandidateSet:
        t0 = now_ms()
        if self.mode == "chat":
            cands = await self._predict_chat(history, latest_partial, request_id)
        else:
            cands = await self._predict_raw(history, latest_partial, request_id)

        cs = CandidateSet(
            request_id=request_id,
            input_latest=latest_partial,
            created_t_ms=t0,
            candidates=cands,
            wall_ms=now_ms() - t0,
        )
        speakables = [c.text for c in cands if c.speakable and c.text]
        cs.agree_len = agreement_len(speakables)
        best = cs.best
        cs.confidence = confidence(best.mean_logprob if best else None, cs.agree_len) if best else 0.0
        return cs

    async def close(self) -> None:
        await self._client.aclose()

    # ---------- raw completion (base/CPTモデル向け・本命) ----------

    def build_raw_prompt(self, history: str, latest_partial: str) -> str:
        parts = []
        if self.preamble:
            parts.append(self.preamble.rstrip() + "\n")
        if history:
            parts.append(history.rstrip() + "\n")
        parts.append(latest_partial)  # 必ずpartialの最終文字でpromptを終える
        return "".join(parts)

    async def _predict_raw(self, history: str, latest_partial: str, request_id: int) -> list[Candidate]:
        prompt = self.build_raw_prompt(history, latest_partial)
        jobs = []
        for i in range(self.n_candidates):
            greedy = i == 0
            jobs.append(self._completion_once(
                prompt=prompt,
                slot=i,
                seed=request_id * 100 + i,
                temperature=0.0 if greedy else self.temperature,
                source="greedy" if greedy else "sample",
                latest_partial=latest_partial,
            ))
        results = await asyncio.gather(*jobs, return_exceptions=True)
        cands: list[Candidate] = []
        for r in results:
            if isinstance(r, BaseException):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                cands.append(Candidate(text="", raw_text=f"<error:{r}>", source="error",
                                       speakable=False, flags=["backend_error"]))
            else:
                cands.append(r)
        return cands

    async def _completion_once(
        self, prompt: str, slot: int, seed: int, temperature: float, source: str, latest_partial: str
    ) -> Candidate:
        t0 = now_ms()
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": self.n_predict,
            "temperature": temperature,
            "top_p": self.top_p,
            "seed": seed,
            "stop": DEFAULT_STOP,
            "cache_prompt": True,
            "id_slot": slot,
            "n_probs": self.n_probs,
        }
        # 反復退化対策(既定はDRY: 「ああああ…」型ループのみ罰し、助詞等の正常な繰り返しは罰しない)
        if self.repeat_penalty > 1.0:
            payload["repeat_penalty"] = self.repeat_penalty
            payload["repeat_last_n"] = 128
        if self.dry_multiplier > 0:
            payload["dry_multiplier"] = self.dry_multiplier
            payload["dry_base"] = 1.75
            payload["dry_allowed_length"] = 2
        resp = await self._client.post(f"{self.base_url}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        gen_ms = now_ms() - t0

        raw = data.get("content", "")
        verdict = validate(raw, latest_partial, self.max_chars, lang=self.lang)
        timings = data.get("timings", {}) or {}
        return Candidate(
            text=verdict.text,
            raw_text=raw,
            source=source,
            speakable=verdict.speakable,
            silence=verdict.silence,
            flags=verdict.flags,
            mean_logprob=_mean_logprob(data),
            gen_ms=gen_ms,
            prompt_ms=timings.get("prompt_ms"),
            predict_ms=timings.get("predicted_ms"),
            prompt_n=timings.get("prompt_n"),
        )

    # ---------- chat completions (instructモデル比較用) ----------

    async def _predict_chat(self, history: str, latest_partial: str, request_id: int) -> list[Candidate]:
        t0 = now_ms()
        user = (
            f"<直前までの発話>\n{history or 'なし'}\n</直前までの発話>\n"
            f"<現在の発話途中>\n{latest_partial}\n</現在の発話途中>\n"
            "この直後に同じ話者が言い足す短い続きだけを出力:"
        )
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "max_tokens": 48,
            "temperature": 0.2,
        }
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        raw = raw.split("</think>")[-1].strip()  # 思考モデルの残骸を除去
        verdict = validate(raw, latest_partial, self.max_chars, lang=self.lang)
        return [Candidate(
            text=verdict.text,
            raw_text=raw,
            source="chat",
            speakable=verdict.speakable,
            silence=verdict.silence,
            flags=verdict.flags,
            gen_ms=now_ms() - t0,
        )]


def _mean_logprob(data: dict[str, Any]) -> float | None:
    """completion_probabilities からトークン平均logprobを取る。形式の版差を吸収。"""
    probs = data.get("completion_probabilities")
    if not probs:
        return None
    vals: list[float] = []
    for item in probs:
        if "logprob" in item:
            vals.append(float(item["logprob"]))
        elif "prob" in item:
            p = max(float(item["prob"]), 1e-9)
            vals.append(math.log(p))
        elif item.get("probs"):
            p = max(float(item["probs"][0].get("prob", 0.0)), 1e-9)
            vals.append(math.log(p))
    if not vals:
        return None
    return sum(vals) / len(vals)
