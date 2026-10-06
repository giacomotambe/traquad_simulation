#!/bin/bash
# Convert a traquad URDF to USD with the Isaac Sim URDF importer (floating base, force drives on all joints;
# gains and targets are set by finalize_usd.py or at runtime). Output: <out_dir>/<urdf name>/<urdf name>.usda
# usage: ./import_urdf.sh <robot.urdf> <out_dir>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/isaac.sh" "$HERE/import_urdf.py" --urdf "$(readlink -f "$1")" --out_dir "$(readlink -m "$2")"
