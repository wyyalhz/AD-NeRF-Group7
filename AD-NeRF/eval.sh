#!/usr/bin/env bash
set -e

SUBJECT=${1:-Obama}
METRICS=${2:-all}

docker compose run --rm web conda run -n adnerf_pre \
  python AD-NeRF/evaluation/evaluate.py --subject "$SUBJECT" --metrics "$METRICS"
