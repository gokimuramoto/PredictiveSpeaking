"""英語テキストの補助判定(en版のフィラー・完結・長さ調整)。

日本語(ja.py)との構造差:
- 単語がスペース区切り → 縫合・切断は単語境界で行える(日本語より単純で安全)
- 助詞縫合は不要。フィラーは um/uh/like 系
"""

from __future__ import annotations

import re
import unicodedata

FILLERS = ("um", "uh", "er", "ah", "like", "you know", "i mean", "well", "so", "kind of")
PARTICLES: set[str] = set()  # 英語に助詞縫合は不要(ja互換のためのプレースホルダ)

_COMPLETE_END = re.compile(r"[.!?]\s*$")
_INCOMPLETE_END = re.compile(
    r"\b(and|or|but|the|a|an|to|of|in|on|at|with|for|that|which|is|are|was|were|be|so|because|if|when|while)\s*$",
    re.I)


def normalize(text: str) -> str:
    """NFKC + 空白圧縮(スペースは保持する — 日本語版と異なる)。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def normalize_for_match(text: str) -> str:
    """一致率計測用: 小文字化し句読点・空白を除去。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s.,!?;:'\"()\-…]", "", t)


def is_incomplete(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    if _COMPLETE_END.search(t):
        return False
    if _INCOMPLETE_END.search(t):
        return True
    return True  # 句点なしは未完結寄り


def strip_trailing_fillers(text: str) -> str:
    t = text
    while True:
        base = t.rstrip(" ,.")
        low = base.lower()
        matched = None
        for f in sorted(FILLERS, key=len, reverse=True):
            if low.endswith(" " + f) or low == f:
                matched = f
                break
        if matched is None:
            return t
        t = base[: len(base) - len(matched)].rstrip(" ,")
        if not t:
            return ""


def strip_leading_fillers(text: str) -> str:
    t = text
    while True:
        base = t.lstrip(" ,.")
        low = base.lower()
        matched = None
        for f in sorted(FILLERS, key=len, reverse=True):
            # Azureはフィラー直後にカンマを挿入する("um, I think")ため両方見る
            if low.startswith(f + " ") or low.startswith(f + ",") or low == f:
                matched = f
                break
        if matched is None:
            return base if base != text else t
        t = base[len(matched):].lstrip(" ,")


def natural_cut(text: str, min_keep: int = 8) -> str:
    """単語境界+接続語を避けた自然な位置まで末尾を戻す。"""
    t = text.rstrip()
    # 途中で切れた最後の単語を落とす(スペースがあれば)
    if " " in t:
        last_space = t.rfind(" ")
        if len(t) - last_space <= 12 and not _COMPLETE_END.search(t):
            cand = t[:last_space]
            if len(cand) >= min_keep:
                t = cand
    # 末尾が接続詞・冠詞なら更に1語戻す
    for _ in range(2):
        if _INCOMPLETE_END.search(t) and " " in t:
            cand = t[: t.rfind(" ")]
            if len(cand) >= min_keep:
                t = cand
            else:
                break
        else:
            break
    return t


def fit_length(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return natural_cut(text[:max_chars])
