#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH="$PWD:$PWD/vendor/florence_only" "${PYTHON_BIN:-python3}" -s -m unittest discover -s tests -v
PYTHONPATH="$PWD/vendor/florence_only" "${PYTHON_BIN:-python3}" -s -m unittest discover -s vendor/florence_only -p test_core.py -v
