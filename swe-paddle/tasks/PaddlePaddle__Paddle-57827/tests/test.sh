#!/usr/bin/env bash
set -uo pipefail

# Target tests for PaddlePaddle__Paddle-57827.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# Nodes are selected by class / generated test id so pytest does not collect
# the module-level helpers `test_ast_only` / `test_legacy_and_pir` imported by
# the test file. Every block runs even if an earlier one fails; the script
# exits non-zero if any block fails.
F=test/dygraph_to_static/test_build_strategy.py
status=0

# P2P tests (pass-to-pass)
python -m pytest \
  "$F::TestResnetWithPass::test_resnet__ast_legacy_ir" \
  "$F::TestError" \
  -q || status=1

# F2P tests (fail-to-pass). Before the fix these abort natively, so each node
# runs in its own process to keep the other results.
for node in \
  TestResnetWithPass::test_resnet__ast_pir_exe \
  TestResnetWithPass::test_in_static_mode_mkldnn__ast_pir_exe; do
  python -m pytest "$F::$node" -q || status=1
done

exit "$status"
