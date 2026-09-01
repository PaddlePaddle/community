#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_lr_scheduler.py::TestCosineAnnealingWarmRestarts::test_CosineRestartsLR test/legacy_test/test_lr_scheduler.py::TestLRSchedulerWithOptimizerArg -q
