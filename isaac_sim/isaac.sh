#!/bin/bash
# Run an Isaac Sim python script headless.
# - Standalone Isaac Sim 6.1 (ISAAC_SIM_DIR, if present): clean environment, no conda variables and Isaac's own
#   CUDA runtime first (the system libcudart 12.0 makes Isaac exit at startup).
# - Otherwise the python of the active environment, if it has Isaac Sim installed with pip (e.g. conda env_isaaclab3).
# usage: ./isaac.sh <script.py> [args...]      ISAAC_SIM_DIR overrides the install path
ISAAC_SIM_DIR=${ISAAC_SIM_DIR:-$HOME/Downloads/isaac-sim-standalone-6.1.0-linux-x86_64}
SCRIPT=$(readlink -f "$1"); shift
if [ ! -x "$ISAAC_SIM_DIR/python.sh" ]; then
  # the scripts use the Isaac Sim 6.x API
  if python -c "import importlib.metadata as m, sys; sys.exit(int(m.version('isaacsim').split('.')[0]) < 6)" 2>/dev/null; then
    exec env OMNI_KIT_ACCEPT_EULA=YES python "$SCRIPT" "$@"
  fi
  echo "Isaac Sim 6.x not found: no $ISAAC_SIM_DIR/python.sh and no isaacsim>=6 in the active python" >&2
  exit 1
fi
cd "$ISAAC_SIM_DIR" && exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin \
  OMNI_KIT_ACCEPT_EULA=YES \
  LD_LIBRARY_PATH="$ISAAC_SIM_DIR/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_runtime/lib" \
  ./python.sh "$SCRIPT" "$@"
