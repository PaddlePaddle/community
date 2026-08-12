#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/ai_edited_test/test_ai_dataloader.py::TestTensorDataset test/legacy_test/test_paddle_utils_data.py::TestAlias::test_compatibility -q
