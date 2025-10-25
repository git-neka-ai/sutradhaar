#!/usr/bin/env bash

set -euo pipefail

# orion: Build a single-file .pyz using shiv by installing the local project (.) so code,
# dependencies, and packaged resources are included. Uses the 'sutradhar' console script
# and a portable "/usr/bin/env python3" shebang. Outputs to dist/sutradhar.pyz.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
DIST_DIR=""${REPO_ROOT}/dist""
DIST_ID=`date +"%y%m%d-%H%M"`
PYZ="${DIST_DIR}/orion-${DIST_ID}.pyz"


mkdir -p "${DIST_DIR}"
rm -f "${PYZ}"

if ! command -v shiv >/dev/null 2>&1; then
  echo "Error: shiv is not installed. Install it with: pip install shiv" >&2
  exit 1
fi

# Install the local project into the pyz so that its code and declared dependencies
# (from pyproject.toml [project.dependencies]) are bundled by shiv.
shiv -c orion -p "/usr/bin/env python3" -o "${PYZ}" "${REPO_ROOT}"

echo "Built ${PYZ}"

if [[ -L "${DIST_DIR}/orion.pyz" ]]; then
  rm "${DIST_DIR}/orion.pyz"
  ln -s "${PYZ}" "${DIST_DIR}/orion.pyz"
  echo "Updated symlink: ${DIST_DIR}/orion.pyz -> ${PYZ}"
fi
