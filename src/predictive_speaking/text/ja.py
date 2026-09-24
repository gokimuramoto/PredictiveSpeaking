"""日本語テキストの補助判定(正規化・文の未完結判定・フィラー・長さ調整)。

ここは「シグナル生成」であり、話す/話さないの最終判断はゲート側のpolicyが行う。
"""

from __future__ import annotations

import re
import unicodedata

# 文末として完結とみなすパターン
_COMPLETE_END = re.compile(
    r"(?:[。！？!?]|です|ます|ました|でした|ません|でしょう|ください|だ|である|よね|ですね|ますね)$"
)

# 末尾がこれらなら明確に「続きがある」
_INCOMPLETE_END = re.compile(
    r"(?:[はがをにでとのへもや]|から|まで|より|って|という|といった|ので|けど|けれど|が、|し、|て|で|には|では|ですが|ますが|および|または|、)$"
)

FILLERS = ("えっと", "えーと", "あのー", "あの", "そのー", "なんか", "なんていうか", "えー", "うーん", "まあ", "ええと")

_PUNCT = re.compile(r"[\s、。！？!?.,・…「」『』()（）　]")


def normalize(text: str) -> str:
    """NFKC正規化 + 空白除去。予測入力・比較の共通前処理。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def normalize_for_match(text: str) -> str:
    """一致率計測用: 正規化に加えて句読点・記号も除去する。"""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


def is_incomplete(text: str) -> bool:
    """発話が文として未完結か。詰まり介入の前提条件シグナル。"""
    t = normalize(text)
    if not t:
        return False
    if _COMPLETE_END.search(t):
        return False
    if _INCOMPLETE_END.search(t):
        return True
    # 明確な手掛かりがない場合、句点なしで終わっていれば未完結寄りに倒す
    return True


def trailing_filler(text: str) -> str | None:
    """末尾のフィラーを返す(なければNone)。"""
    t = normalize(text)
    for f in sorted(FILLERS, key=len, reverse=True):
        if t.endswith(f):
            return f
    return None


def strip_trailing_fillers(text: str) -> str:
    """末尾のフィラー(連続・読点混じり含む)を取り除く。

    「〜についてえっとあの」→「〜について」。予測をフィラーに条件付けさせない
    ための前処理。全部フィラーの場合は空文字を返す(呼び出し側で送信スキップ)。
    """
    t = text
    while True:
        base = t.rstrip("、。 　")
        f = trailing_filler(base)
        if f is None:
            return t
        t = base[: len(base) - len(f)]


PARTICLES = set("はがをにでとのへもや")


def dedup_boundary_particle(prev_text: str, next_text: str) -> str:
    """つなぎ目の助詞重複を縫合する。

    prev末尾とnext先頭が同一の助詞なら、nextから1つ落とす(「〜を」+「を報告」→「報告」)。
    完全一致の助詞1文字に限定するため誤削除しない。読点も同様に扱う。
    戻り値: 縫合後のnext_text。
    """
    p = normalize_for_match(prev_text)
    if not p or not next_text:
        return next_text
    nxt = next_text
    # next先頭の読点は接合時に不要
    while nxt[:1] == "、" and p:
        nxt = nxt[1:]
    if nxt[:1] in PARTICLES and p[-1:] == nxt[:1]:
        nxt = nxt[1:].lstrip("、")
    return nxt


def natural_cut(text: str, min_keep: int = 4) -> str:
    """末尾を自然な切れ目(助詞・読点・文末表現)まで戻す。境界がなければそのまま。"""
    for i in range(len(text), min_keep - 1, -1):
        head = text[:i]
        if _INCOMPLETE_END.search(head) or _COMPLETE_END.search(head):
            return head
    return text


def strip_leading_fillers(text: str) -> str:
    """先頭のフィラー(読点付き含む)を除去する。候補が「えっと、〜」で始まるのを防ぐ。"""
    t = text
    while True:
        base = t.lstrip("、。・… 　")
        matched = None
        for f in sorted(FILLERS, key=len, reverse=True):
            if base.startswith(f):
                matched = f
                break
        if matched is None:
            return base if base != text else t
        t = base[len(matched):]


def fit_length(text: str, max_chars: int) -> str:
    """発話用に長さを調整する。可能なら助詞・読点境界で切り、なければ強制切断。"""
    if len(text) <= max_chars:
        return text
    return natural_cut(text[:max_chars])
