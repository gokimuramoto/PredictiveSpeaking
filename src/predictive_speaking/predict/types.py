"""予測結果の型。予測エンジンは判断せず、シグナル付き候補セットを出力するだけ。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    text: str                    # 検証・長さ調整済みテキスト("" = 沈黙/不成立)
    raw_text: str                # モデル生出力
    source: str                  # "greedy" | "sample" | "chat" など
    speakable: bool
    silence: bool = False
    flags: list[str] = field(default_factory=list)
    mean_logprob: float | None = None
    gen_ms: float = 0.0          # この候補の生成所要(壁時計)
    prompt_ms: float | None = None
    predict_ms: float | None = None
    prompt_n: int | None = None  # プリフィルしたトークン数(KVキャッシュ効果の観測用)


@dataclass
class CandidateSet:
    request_id: int
    input_latest: str            # 予測時点の最新partial(由来prefix)
    created_t_ms: float
    candidates: list[Candidate] = field(default_factory=list)
    wall_ms: float = 0.0         # セット全体の壁時計時間
    agree_len: int = 0           # speakable候補間の先頭一致長
    confidence: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Candidate | None:
        """greedy優先で最初のspeakable候補。"""
        ordered = sorted(self.candidates, key=lambda c: 0 if c.source == "greedy" else 1)
        for c in ordered:
            if c.speakable and c.text:
                return c
        return None
