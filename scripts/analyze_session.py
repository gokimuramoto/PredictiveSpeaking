"""実セッションログの事後分析。

candidatesイベントの各候補を、その後実際に確定した発話(asr_final)と突き合わせて
自動採点する。ゲート発火(assist_play)時に何が鳴ったかも列挙する。

  uv run python scripts/analyze_session.py logs/app_session.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_speaking.score import leading_match_len, percentile  # noqa: E402
from predictive_speaking.text.ja import normalize_for_match  # noqa: E402


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def truth_for(partial: str, final_text: str) -> str | None:
    """partialに対する正解の続きをfinalから取る(正規化prefix一致で位置合わせ)。"""
    np_, nf = normalize_for_match(partial), normalize_for_match(final_text)
    if not np_ or not nf.startswith(np_):
        return None
    return nf[len(np_):]


def main(path: str) -> None:
    events = load(path)
    utterances: list[dict] = []   # {final, candidates:[{input,best,conf,t}], plays:[...]}
    current_cands: list[dict] = []
    current_plays: list[dict] = []
    history_len = 0
    rows = []
    plays_all = []
    rewinds = 0
    asr_partials = 0

    for ev in events:
        e = ev.get("event")
        if e == "candidates" and ev.get("best"):
            current_cands.append(ev)
        elif e == "asr_partial":
            asr_partials += 1
        elif e == "asr_rewind":
            rewinds += 1
        elif e == "assist_play":
            current_plays.append(ev)
        elif e == "asr_final":
            final = ev.get("text", "")
            for c in current_cands:
                t = truth_for(c["input"], final)
                if t is None:
                    continue
                rows.append({
                    "input": c["input"], "best": c["best"], "truth": t,
                    "match": leading_match_len(c["best"], t) if t else 0,
                    "conf": c.get("confidence", 0), "history_len": history_len,
                    "empty_truth": len(t) == 0,
                })
            for p in current_plays:
                plays_all.append({**p, "final": final})
            utterances.append({"final": final, "n_cands": len(current_cands)})
            current_cands, current_plays = [], []
            history_len += 1

    print(f"== {path} ==")
    print(f"発話数: {len(utterances)}  partial: {asr_partials}  巻き戻り: {rewinds}")
    scored = [r for r in rows if not r["empty_truth"]]
    if scored:
        hit = lambda k: sum(1 for r in scored if r["match"] >= k) / len(scored)
        print(f"採点対象候補: {len(scored)}  hit@1: {hit(1):.3f}  hit@2: {hit(2):.3f}")
        confs = sorted(scored, key=lambda r: -r["conf"])
        top = confs[: max(1, len(confs) // 4)]
        print(f"conf上位25%: hit@1 {sum(1 for r in top if r['match']>=1)/len(top):.3f}")
        thin = [r for r in scored if r["history_len"] <= 2]
        rich = [r for r in scored if r["history_len"] > 2]
        if thin and rich:
            print(f"履歴≤2文: hit@1 {sum(1 for r in thin if r['match']>=1)/len(thin):.3f} (n={len(thin)})"
                  f" / 履歴>2文: hit@1 {sum(1 for r in rich if r['match']>=1)/len(rich):.3f} (n={len(rich)})")
        # 文末で発話が終わっていた(=正解が空)割合
    empty = sum(1 for r in rows if r["empty_truth"])
    print(f"「続き無し」位置(発話がそこで終了)の候補: {empty}/{len(rows)}")

    print(f"\n介入(assist_play): {len(plays_all)}回")
    for p in plays_all[:10]:
        print(f"  再生: '{p.get('text')}' | その発話の確定文: {p.get('final','')[:40]}")

    print("\n-- 外れ例(match=0, conf高い順に8件):")
    for r in sorted(scored, key=lambda r: -r["conf"])[:40]:
        if r["match"] == 0:
            print(f"  …{r['input'][-16:]} | 予測: {r['best']} | 実際: {r['truth'][:16]}")
    print("\n-- 当たり例(match>=2, 8件):")
    n = 0
    for r in scored:
        if r["match"] >= 2 and n < 8:
            print(f"  …{r['input'][-16:]} | 予測: {r['best']} | 実際: {r['truth'][:16]}")
            n += 1


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/app_session.jsonl")
