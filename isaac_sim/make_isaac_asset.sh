#!/bin/bash
# Build the instanceable Isaac Sim / Isaac Lab asset of traquad from the current xacro:
# xacro -> URDF (host mesh paths) -> USD (URDF importer) -> default physics settings (finalize_usd.py).
# usage: ./make_isaac_asset.sh [roller_damping] [roller_friction]            output: assets/traquad/traquad.usda
#        TRACK_MODEL=cylinder ./make_isaac_asset.sh                        output: assets/traquad_cylinder/traquad.usda
#   extra finalize_usd.py options can be passed in FINALIZE_ARGS (e.g. "--wheel_max_torque 0.5")
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
TRACK_MODEL=${TRACK_MODEL:-rollers}
NAME=traquad; [ "$TRACK_MODEL" = cylinder ] && NAME=traquad_cylinder
TMP=$(mktemp -d)
TRACK_MODEL=$TRACK_MODEL "$HERE/make_urdf.sh" "$TMP/traquad.urdf"
"$HERE/import_urdf.sh" "$TMP/traquad.urdf" "$TMP/usd" > "$TMP/import.log" 2>&1
grep -a "Import complete" "$TMP/import.log"
rm -rf "$HERE/assets/$NAME"
mv "$TMP/usd/traquad" "$HERE/assets/$NAME"
"$HERE/isaac.sh" "$HERE/finalize_usd.py" --usd "$HERE/assets/$NAME/traquad.usda" \
  --roller_damping "${1:-1e-4}" --roller_friction "${2:-0.06}" $FINALIZE_ARGS > "$TMP/finalize.log" 2>&1
grep -a "^finalized\|Traceback" "$TMP/finalize.log"
rm -rf "$TMP"
