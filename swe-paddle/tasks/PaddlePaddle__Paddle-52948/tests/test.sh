#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-52948 (#52948 + #53572).
# Run from the root of a PaddlePaddle/Paddle checkout with Paddle importable.
#
# Do NOT pass python/paddle/... paths to pytest from the repo root: pytest then
# nests imports as python.paddle and can re-register VarBase against the
# installed paddle (double registration). Run the fluid unittest from its
# own directory as a flat module instead.

python -m pytest test/dygraph_to_static/test_tensor_hook.py -q

(
  cd python/paddle/fluid/tests/unittests
  # Prefer importlib mode so parents of this dir are not added as packages.
  if python -m pytest --help 2>/dev/null | grep -q -- '--import-mode'; then
    python -m pytest test_tensor_register_hook.py -q --import-mode=importlib
  else
    python -m unittest test_tensor_register_hook
  fi
)
