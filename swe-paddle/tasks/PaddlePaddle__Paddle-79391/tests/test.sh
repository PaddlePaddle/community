#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Every module below mutates process-global state: sys.meta_path (the torch
# proxy finder), sys.modules["torch*"], sys.path (fake torch package) and,
# under level 2, the paddle namespace itself. Run each module in its own
# interpreter so no module can observe another module's leftovers.

# ---------------------------------------------------------------------------
# P2P (pass-to-pass): the pre-existing torch import-proxy behaviour. The gold
# patch touches paddle/compat/proxy.py, so these must keep passing before and
# after the fix. Neither file is modified by the gold patch or the test patch.
# ---------------------------------------------------------------------------
"$PYTHON_BIN" -m pytest -q test/compat/test_torch_proxy.py
"$PYTHON_BIN" -m pytest -q test/compat/test_torch_proxy_mixed.py

# ---------------------------------------------------------------------------
# F2P (fail-to-pass): run both target modules even when the first one fails, so
# a Base run records the complete role matrix before the wrapper exits 1.
# ---------------------------------------------------------------------------
f2p_status=0

if ! "$PYTHON_BIN" -m pytest -q test/compat/test_compat_namespace_aliased.py; then
  f2p_status=1
fi

if ! "$PYTHON_BIN" -m pytest -q test/compat/test_compat_level2_internal_composites.py; then
  f2p_status=1
fi

exit "$f2p_status"
