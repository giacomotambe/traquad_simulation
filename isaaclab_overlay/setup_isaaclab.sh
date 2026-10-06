#!/bin/bash
# Isaac Lab 3.0 with the TraQuad tasks.
# Clones Isaac Lab (release/3.0.0, the commit used for the TraQuad work) into ../isaaclab_traquad if it is not there,
# then links the TraQuad files of this folder into it. The asset is found through the repository layout
# (traquad_simulation/isaaclab_traquad/...), so keep isaaclab_traquad inside traquad_simulation.
# usage: ./isaaclab_overlay/setup_isaaclab.sh      then, in the conda env: cd isaaclab_traquad && ./isaaclab.sh -i
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
LAB=$(dirname "$HERE")/isaaclab_traquad
COMMIT=7aba91f629128e572b53179b2a34c0bbf25ae346
if [ ! -d "$LAB/.git" ]; then
  git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git "$LAB"
  git -C "$LAB" checkout -B traquad "$COMMIT"
fi
for f in source/isaaclab_assets/isaaclab_assets/robots/traquad.py source/isaaclab_tasks/isaaclab_tasks/contrib/traquad; do
  if [ -e "$LAB/$f" ] && [ ! -L "$LAB/$f" ]; then
    echo "$LAB/$f exists and is not a link: move it away first" >&2
    exit 1
  fi
  ln -sfn "$HERE/$f" "$LAB/$f"
  echo "linked $f"
done
