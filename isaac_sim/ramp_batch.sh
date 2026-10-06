#!/bin/bash
# Side-slope matrix: roller dry friction x slope angle (+ cylindrical wheels as reference). Results in ./results
# usage: ./ramp_batch.sh <rollers.usda> [cylinders.usda]      MASS (default 7.8) and DAMPING (default 1e-4) via env
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/results
mkdir -p "$OUT"
MASS=${MASS:-7.8}
DAMPING=${DAMPING:-1e-4}
ANGLES="5 10 15 20 25 30 35 40"
if [ -n "$2" ]; then
  echo "===== cylinders $(date +%T)"
  "$HERE/isaac.sh" "$HERE/ramp_test.py" --usd "$(readlink -f "$2")" --mass $MASS --angles $ANGLES \
    > "$OUT/ramp_cylinders.log" 2>&1
  grep -a "^RES\|Traceback" "$OUT/ramp_cylinders.log"
fi
for tau in ${TAUS:-0.0 0.003 0.006 0.012 0.018 0.03 0.06}; do
  echo "===== rollers friction $tau $(date +%T)"
  "$HERE/isaac.sh" "$HERE/ramp_test.py" --usd "$(readlink -f "$1")" --mass $MASS --roller_damping $DAMPING \
    --roller_friction $tau --angles $ANGLES > "$OUT/ramp_rollers_$tau.log" 2>&1
  grep -a "^RES\|Traceback" "$OUT/ramp_rollers_$tau.log"
done
