#!/bin/bash
# Build the instanceable Isaac Sim / Isaac Lab asset of traquad from the current xacro:
# xacro -> URDF (host mesh paths) -> USD (URDF importer) -> default physics settings (finalize_usd.py).
# usage: ./make_isaac_asset.sh [roller_damping] [roller_friction]            output: assets/traquad/traquad.usda
#   extra finalize_usd.py options can be passed in FINALIZE_ARGS (e.g. "--hfe_offset -0.05")
#   finalize_usd.py alone can be rerun on the existing asset to change drives and frictions (no ROS needed)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=traquad
TMP=$(mktemp -d)
"$HERE/make_urdf.sh" "$TMP/traquad.urdf"
"$HERE/import_urdf.sh" "$TMP/traquad.urdf" "$TMP/usd" > "$TMP/import.log" 2>&1
grep -a "Import complete" "$TMP/import.log"
rm -rf "$HERE/assets/$NAME"
mv "$TMP/usd/traquad" "$HERE/assets/$NAME"
ROLLER_ARGS=""
[ -n "$1" ] && ROLLER_ARGS="$ROLLER_ARGS --roller_damping $1"
[ -n "$2" ] && ROLLER_ARGS="$ROLLER_ARGS --roller_friction $2"
"$HERE/isaac.sh" "$HERE/finalize_usd.py" --usd "$HERE/assets/$NAME/traquad.usda" $ROLLER_ARGS $FINALIZE_ARGS \
  > "$TMP/finalize.log" 2>&1
grep -a "^finalized\|Traceback" "$TMP/finalize.log"
rm -rf "$TMP"
