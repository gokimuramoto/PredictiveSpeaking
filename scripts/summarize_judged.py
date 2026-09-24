"""ジャッジ済みJSONLの再集計(再ジャッジなし)。pplgapとの相関も出す。"""

from __future__ import annotations

import json
import sys


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


for path in sys.argv[1:]:
    rows = [r for r in load(path) if r.get("coherence") is not None]
    n = len(rows)
    coh = [r["coherence"] for r in rows]
    inte = [r["intent"] for r in rows]
    print(f"== {path} (n={n}) ==")
    print(f"  逸脱率(coherence=0): {sum(1 for c in coh if c==0)/n:.3f}")
    print(f"  coherence平均: {sum(coh)/n:.2f}")
    print(f"  意図一致率(>=1): {sum(1 for i in inte if i>=1):.0f}/{n} = {sum(1 for i in inte if i>=1)/n:.3f}")
    print(f"  ゲート試算: confidence上位50%に限定した場合")
    top = sorted(rows, key=lambda r: r["confidence"], reverse=True)[: n // 2]
    print(f"    逸脱率: {sum(1 for r in top if r['coherence']==0)/len(top):.3f}"
          f"  意図一致(>=1): {sum(1 for r in top if r['intent']>=1)/len(top):.3f}")
    if any("pplgap" in r for r in rows):
        with_gap = [r for r in rows if "pplgap" in r]
        derail = [r["pplgap"] for r in with_gap if r["coherence"] == 0]
        ok = [r["pplgap"] for r in with_gap if r["coherence"] == 2]
        if derail and ok:
            print(f"  pplgap平均: coherence=0群 {sum(derail)/len(derail):+.3f} vs coherence=2群 {sum(ok)/len(ok):+.3f}")
