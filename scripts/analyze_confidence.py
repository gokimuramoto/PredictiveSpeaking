"""既存ベンチ行データから、確信度シグナルで条件付けした精度を分析する。

「常時予測し、確信が高い瞬間だけ介入する」設計の妥当性検証:
シグナル(confidence / agree_len)の上位に絞ったとき hit@1 がどこまで上がるか。
"""

from __future__ import annotations

import json
import sys


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r["speakable"] and r["pred"]]


def hit_rate(rows: list[dict], k: int = 1) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["match_len"] >= k) / len(rows)


def analyze(path: str) -> None:
    rows = load(path)
    print(f"\n### {path}  (speakable rows: {len(rows)})")
    print(f"  all: hit@1={hit_rate(rows,1):.3f} hit@2={hit_rate(rows,2):.3f}")

    # confidence分位
    by_conf = sorted(rows, key=lambda r: r["confidence"], reverse=True)
    for frac, name in [(0.25, "top25%"), (0.5, "top50%")]:
        sub = by_conf[: max(1, int(len(rows) * frac))]
        thr = sub[-1]["confidence"]
        print(f"  conf {name} (>= {thr:.2f}): n={len(sub)} hit@1={hit_rate(sub,1):.3f} hit@2={hit_rate(sub,2):.3f}")

    # agree_len(候補間一致)しきい値
    for thr in [2, 3, 5]:
        sub = [r for r in rows if r["agree_len"] >= thr]
        if sub:
            print(f"  agree_len>={thr}: n={len(sub)} ({len(sub)/len(rows):.0%}) hit@1={hit_rate(sub,1):.3f} hit@2={hit_rate(sub,2):.3f}")

    # 例: 高確信のhitとmiss
    hits = [r for r in by_conf if r["match_len"] >= 2][:3]
    misses = [r for r in by_conf if r["match_len"] == 0][:3]
    print("  -- 高確信hit例:")
    for r in hits:
        print(f"     …{r['input_latest'][-14:]} | 予測: {r['pred']} | 実際: {r['truth'][:14]}…")
    print("  -- 高確信miss例:")
    for r in misses:
        print(f"     …{r['input_latest'][-14:]} | 予測: {r['pred']} | 実際: {r['truth'][:14]}…")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
