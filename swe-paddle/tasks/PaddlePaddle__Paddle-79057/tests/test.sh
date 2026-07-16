#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_restricted_unpickler_mro.py -q
