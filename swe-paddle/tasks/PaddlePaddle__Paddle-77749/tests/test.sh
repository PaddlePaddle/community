#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# P2P (pass-to-pass): existing RNN utility regression guard. The gold patch
# does not modify test/legacy_test/test_rnn_cell_api.py, so this node must pass
# both before and after the solution is applied.
(
    cd test/legacy_test
    "${PYTHON_BIN}" -m pytest -q \
        test_rnn_cell_api.py::TestRnnUtil::test_case
)

# F2P (fail-to-pass): all 21 cases in the newly added API test file. They run
# under pytest so every case is collected and reported as its own node. The
# target APIs are imported inside the test bodies, so on the base revision the
# file still imports and each case fails individually instead of aborting
# collection.
"${PYTHON_BIN}" -m pytest -q test/legacy_test/test_rnn_utils.py
