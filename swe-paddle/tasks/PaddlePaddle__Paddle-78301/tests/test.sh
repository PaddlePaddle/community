#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/test/legacy_test:$(pwd)/test/dygraph_to_static${PYTHONPATH:+:$PYTHONPATH}"
export FLAGS_enable_pir_api=0

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_api_compatibility_part3.py::TestLayerAndTensorToAPI \
  -q

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_api_compatibility_part3.py::TestTensorToCopyCompatibility::test_copy_as_positional_argument \
  -q

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_base_layer.py::TestLayerTo::test_main \
  -q
