"""piper.train 系モジュールを torch.load の safe_globals 許可付きで起動するラッパー。

PyTorch 2.6+ は torch.load(weights_only=True) が既定/明示で、Lightning CLI が
--ckpt_path を検証する際に lessac チェックポイント内の pathlib.PosixPath を拒否する
(UnpicklingError: Unsupported global pathlib.PosixPath)。rhasspy公式配布物なので
信頼できるとみなし、PosixPath 系を許可リストに入れてから本体を実行する。

  python piper_wrap.py piper.train fit --data.voice_name ...
  python piper_wrap.py piper.train.export_onnx --checkpoint ... --output-file ...
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import torch

torch.serialization.add_safe_globals([
    pathlib.PosixPath, pathlib.PurePosixPath, pathlib.Path, pathlib.PurePath,
])

# PyTorch 2.9+ の torch.onnx.export は既定で dynamo エクスポータになり、piper の VITS
# (weight_norm・動的長)を torch.export でトレースできず失敗する。piper の export_onnx.py は
# dynamo 引数を渡さないので、ここで従来(TorchScript)エクスポータを既定にする。
_orig_onnx_export = torch.onnx.export


def _legacy_onnx_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _orig_onnx_export(*args, **kwargs)


torch.onnx.export = _legacy_onnx_export

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: piper_wrap.py <module> [args...]", file=sys.stderr)
        sys.exit(2)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
