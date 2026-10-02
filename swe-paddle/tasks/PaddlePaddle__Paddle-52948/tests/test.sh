#!/usr/bin/env bash
set -uo pipefail

# Target tests for PaddlePaddle__Paddle-52948 (#52948 + #53572).
# Run from the root of a PaddlePaddle/Paddle checkout with Paddle importable.
#
# Run each test file through its own unittest entry. Collecting the fluid test
# with pytest from the source tree walks the python/ package __init__.py chain,
# loading Paddle as both `paddle` and `python.paddle` and re-initializing the
# native module. Both files run even if the first fails, so Base logs keep the
# full result set.

status=0

python test/dygraph_to_static/test_tensor_hook.py || status=1
python python/paddle/fluid/tests/unittests/test_tensor_register_hook.py || status=1

exit "$status"
