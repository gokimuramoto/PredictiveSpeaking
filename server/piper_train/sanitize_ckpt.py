"""旧 rhasspy/piper チェックポイントを piper1-gpl の LightningCLI で使えるように整形する。

LightningCLI は --ckpt_path の hyper_parameters をCLI引数として再解析するため、
旧版固有のキー(model.sample_bytes 等)で "Parsing of ckpt_path hyperparameters failed" になる。
hyper_parameters を取り除き(内容は参考用にjson保存)、重み・optimizer・epoch はそのまま残す。

  python sanitize_ckpt.py ckpt/lessac_medium.ckpt ckpt/lessac_medium_clean.ckpt
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.PurePosixPath, pathlib.Path])

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
print("top-level keys:", list(ck.keys()))
print("epoch:", ck.get("epoch"), "global_step:", ck.get("global_step"))
hp = ck.get("hyper_parameters", {})
print("hyper_parameters keys:", sorted(str(k) for k in hp.keys()))
with open(dst + ".hparams.json", "w", encoding="utf-8") as fh:
    json.dump({str(k): str(v) for k, v in hp.items()}, fh, indent=1)
ck.pop("hyper_parameters", None)
sd = ck.get("state_dict", {})
print("state_dict tensors:", len(sd), "sample keys:", list(sd.keys())[:5])
torch.save(ck, dst)
print("saved:", dst)
