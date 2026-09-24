"""候補検証。

候補を捨てる/加工するのではなく、原則としてフラグ(シグナル)を付けてゲートに渡す。
ただし「発話として成立しない」もの(空・記号のみ・メタ出力)だけは speakable=False にする。
旧実装 backend/azurePredictor.js / server.js の正規表現群を移植・整理。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .text import ja

SILENCE_TOKEN = "<SILENCE>"

_ASSISTANT_START = re.compile(
    r"^(はい|いいえ|ええ|うん|ううん|そうですね|そうです|わかりました|了解です|承知しました|"
    r"かしこまりました|こちらこそ|ありがとうございます|どういたしまして|失礼しました|すみません|"
    r"申し訳|よろしくお願いします|もちろん|大丈夫です|いいですよ|いいですね|なるほど|どうしたの)"
)
_ASSISTANT_BODY = re.compile(
    r"(どうぞ|続けてください|お手伝い|ご質問|お答え|教えてください|してください|できますか|いたします|お話し|対応します)"
)
_META = re.compile(r"[<>{}\[\]|]|```|出力[:：]|予測[:：]|続き[:：]|例[:：]")
_DIGIT = re.compile(r"[0-9０-９]")
_NEGATION = re.compile(r"(ない|ません|じゃな|ではな|なかった|違う|違います|不可能|できない)")
_QUESTION = re.compile(r"(？|\?|ですか$|ますか$|でしょうか$|かな$|なぜ|どうして)")
_SYMBOL_ONLY = re.compile(r"^[\s、。！？!?.,\-'\"・…]+$")


@dataclass
class Verdict:
    text: str                 # 加工後(長さ調整済み)の候補テキスト
    speakable: bool           # 音声として成立するか(最低限のゲート)
    silence: bool = False     # モデルが明示的に沈黙を選んだ
    flags: list[str] = field(default_factory=list)  # policy用シグナル


def clean_raw(raw: str) -> str:
    """モデル出力からメタ断片・引用符等を除去する(文分割はvalidate側)。"""
    t = ja.normalize(raw)
    t = re.sub(r"\[end of text\]|\[(PAD|UNK|CLS|SEP)\]|</?s>|<\|[^|]*\|>", "", t, flags=re.I)
    t = re.sub(r"[「」『』\"'“”]", "", t)
    t = t.split("\n")[0]
    # 先頭の句読点は表示・発話とも不自然なので除去
    return t.strip().lstrip("、。・…,.")


def _choose_segment(text: str, latest_partial: str = "") -> tuple[str, list[str], bool]:
    """候補に使う文断片を選ぶ。戻り値: (断片, フラグ, 生成打ち切りか)。

    現在の発話が完結していて最初の断片が極短(語尾の残りだけ)なら、次の文の
    内容を候補にする(「〜です」+「ね」だけの無価値候補を避ける)。
    発話が未完結の途中で次の文へ飛ぶと接続が壊れるため、その場合は飛ばない。
    選んだ断片が「文末記号なしで生成が尽きた末尾」なら gen_cut=True。
    """
    ends_with_punct = bool(re.search(r"[。！？!?]\s*$", text))
    segs = [s.strip().lstrip("、・…,.") for s in re.split(r"[。！？!?]", text)]
    segs = [s for s in segs if s]
    if not segs:
        return "", [], False
    partial_complete = bool(latest_partial) and not ja.is_incomplete(latest_partial)
    if len(ja.normalize_for_match(segs[0])) < 4 and len(segs) >= 2 and partial_complete:
        idx = 1
        flags = ["next_sentence"]
    else:
        idx = 0
        flags = []
    gen_cut = (idx == len(segs) - 1) and not ends_with_punct
    return segs[idx], flags, gen_cut


def _is_degenerate(text_norm: str, input_norm: str) -> bool:
    """反復退化の検出: 「ああああ」「マイクテストマイクテスト」「、の、の、の」等。

    短い入力でLMが反復ループに落ちた出力は発話として成立しないため不可とする。
    """
    n = len(text_norm)
    if n == 0:
        return False
    # 文字多様性: 「あああらああぁあい」のような変種混じりの退化を捕捉
    if n >= 6:
        from collections import Counter

        counts = Counter(text_norm)
        if len(counts) <= 2 or counts.most_common(1)[0][1] / n > 0.5:
            return True
    # 内部反復: 1〜6文字の単位の3回以上の繰り返しが全体を占める
    for unit_len in range(1, min(6, n // 2) + 1):
        reps = n // unit_len
        if reps >= 3 and (text_norm[:unit_len] * reps) == text_norm[: unit_len * reps] \
                and unit_len * reps >= n - unit_len:
            return True
    # 入力末尾の語句を2回以上繰り返している
    for tail_len in range(3, 9):
        tail = input_norm[-tail_len:]
        if len(tail) == tail_len and text_norm.count(tail) >= 2:
            return True
    return False


_ASSISTANT_EN = re.compile(
    r"^(sure|of course|certainly|i can|i'd be happy|here('s| is)|as an ai|great question)\b", re.I)


def _validate_en(raw: str, latest_partial: str, max_chars: int) -> Verdict:
    """英語版の検証。日本語版より単純(助詞縫合なし・単語境界処理)。"""
    from .text import en

    t = ja_clean_shared(raw)
    t = en.strip_leading_fillers(t)
    if not t or _SYMBOL_ONLY.match(t):
        return Verdict(text="", speakable=False, flags=["empty"])
    if _META.search(t):
        return Verdict(text=t, speakable=False, flags=["meta"])

    segs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    flags: list[str] = []
    text = segs[0] if segs else ""
    text = text.rstrip(".!?")
    if not text:
        return Verdict(text="", speakable=False, flags=["empty"])
    if _is_degenerate(en.normalize_for_match(text), en.normalize_for_match(latest_partial)):
        return Verdict(text=text, speakable=False, flags=["degenerate"])
    # 単語レベルの反復退化: 「was the the the the and the ...」型。文字単位の
    # _is_degenerateは前置き付き反復をすり抜けるため、支配的な単語の割合で検出する
    words = [w for w in re.split(r"[^a-z']+", text.lower()) if w]
    if len(words) >= 4:
        top = max(words.count(w) for w in set(words))
        if top / len(words) > 0.5:
            return Verdict(text=text, speakable=False, flags=["degenerate"])
    if _ASSISTANT_EN.match(text):
        return Verdict(text=text, speakable=False, flags=["assistant_like"])
    if "?" in text:
        flags.append("question")
    if _DIGIT.search(text):
        flags.append("digits")
    if len(text) > max_chars:
        flags.append("truncated")
        text = en.fit_length(text, max_chars)
    else:
        trimmed = en.natural_cut(text) if not en._COMPLETE_END.search(t) else text
        if trimmed != text and len(trimmed) >= 8:
            flags.append("gen_cut")
            text = trimmed
    # エコー(入力末尾の繰り返し)
    tail = en.normalize_for_match(latest_partial)[-16:]
    if tail and en.normalize_for_match(text) and tail.endswith(en.normalize_for_match(text)[:4]):
        flags.append("echo")
    speakable = "echo" not in flags and len(en.normalize_for_match(text)) >= 3
    return Verdict(text=text, speakable=speakable, flags=flags)


def ja_clean_shared(raw: str) -> str:
    """言語共通のメタ断片除去(NFKC・引用符・制御トークン)。空白は保持。"""
    import unicodedata

    t = unicodedata.normalize("NFKC", raw)
    t = re.sub(r"\[end of text\]|\[(PAD|UNK|CLS|SEP)\]|</?s>|<\|[^|]*\|>", "", t, flags=re.I)
    t = re.sub(r"[「」『』\"'“”]", "", t)
    t = t.split("\n")[0]
    return re.sub(r"\s+", " ", t).strip()


def validate(
    raw: str,
    latest_partial: str,
    max_chars: int | None = None,
    lang: str = "ja",
) -> Verdict:
    if max_chars is None:
        max_chars = 60 if lang == "en" else 22
    if lang == "en":
        return _validate_en(raw, latest_partial, max_chars)
    if ja.normalize(raw) == SILENCE_TOKEN or raw.strip() == SILENCE_TOKEN:
        return Verdict(text="", speakable=False, silence=True, flags=["silence"])

    cleaned = ja.strip_leading_fillers(clean_raw(raw))
    if not cleaned or _SYMBOL_ONLY.match(cleaned):
        return Verdict(text="", speakable=False, flags=["empty"])
    if _META.search(cleaned):
        return Verdict(text=cleaned, speakable=False, flags=["meta"])

    text, flags, gen_cut = _choose_segment(cleaned, latest_partial)
    if not text:
        return Verdict(text="", speakable=False, flags=["empty"])
    if gen_cut:
        # 生成トークン切れの断片は自然な切れ目(助詞・読点等)まで戻す
        trimmed = ja.natural_cut(text)
        if trimmed != text:
            flags = flags + ["gen_cut"]
            text = trimmed
    if _is_degenerate(ja.normalize_for_match(text), ja.normalize_for_match(latest_partial)):
        return Verdict(text=text, speakable=False, flags=["degenerate"])

    # つなぎ目の助詞重複を縫合(「〜を」+候補「を報告します」→「報告します」)
    deduped = ja.dedup_boundary_particle(latest_partial, text)
    if deduped != text:
        flags = flags + ["particle_dedup"]
        text = deduped
        if not text:
            return Verdict(text="", speakable=False, flags=flags + ["empty"])

    # 発話がすでに完結しているのに極短の語尾しか足せない候補は価値がない
    if len(ja.normalize_for_match(text)) <= 3 and not ja.is_incomplete(latest_partial):
        return Verdict(text=text, speakable=False, flags=flags + ["trivial_tail"])

    # 入力末尾の単純な繰り返し(エコー)
    tail = ja.normalize(latest_partial)[-12:]
    if tail and (text in tail or (len(text) >= 2 and tail.endswith(text[:2]) and text in ja.normalize(latest_partial))):
        flags.append("echo")

    if _ASSISTANT_START.match(text) or _ASSISTANT_BODY.search(text):
        flags.append("assistant_like")
    if _DIGIT.search(text):
        flags.append("digits")
    if _NEGATION.search(text):
        flags.append("negation")
    if _QUESTION.search(text):
        flags.append("question")

    if len(text) > max_chars:
        flags.append("truncated")
        text = ja.fit_length(text, max_chars)

    # ひらがな・漢字を全く含まない極端な出力(記号・英数のみ等)は不可
    if not re.search(r"[぀-ゟ一-鿿゠-ヿ]", text):
        return Verdict(text=text, speakable=False, flags=flags + ["non_japanese"])

    speakable = "echo" not in flags and "assistant_like" not in flags
    return Verdict(text=text, speakable=speakable, flags=flags)


def agreement_len(texts: list[str]) -> int:
    """候補間の先頭一致長。分散が大きい=確信度が低い、のシグナル。"""
    norm = [ja.normalize_for_match(t) for t in texts if t]
    if not norm:
        return 0
    if len(norm) == 1:
        return len(norm[0])
    first = norm[0]
    n = min(len(t) for t in norm)
    i = 0
    while i < n and all(t[i] == first[i] for t in norm):
        i += 1
    return i


def confidence(mean_logprob: float | None, agree_len: int) -> float:
    """0..1の簡易確信度。policy調整用の合成シグナル(重みは後で調整)。"""
    p = math.exp(mean_logprob) if mean_logprob is not None else 0.5
    a = min(agree_len, 5) / 5.0
    return round(0.6 * p + 0.4 * a, 3)
