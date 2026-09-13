#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_kv_server.py -q
