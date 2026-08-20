#!/bin/zsh
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd -P)"
PLUGIN_ROOT="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd -P)"
DEFAULT_REPOSITORY="$(cd -- "$PLUGIN_ROOT/../.." && pwd -P)"
CONFIGURED_REPOSITORY="$DEFAULT_REPOSITORY"
RUNTIME_HOME="${VIEWFORGE_LOCAL_RUNTIME_HOME:-$HOME/Library/Application Support/ViewForge 3D Local Plugin}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository)
      [[ $# -ge 2 ]] || { echo "--repository requires a path" >&2; exit 64; }
      CONFIGURED_REPOSITORY="$2"
      shift 2
      ;;
    --runtime-home)
      [[ $# -ge 2 ]] || { echo "--runtime-home requires a path" >&2; exit 64; }
      RUNTIME_HOME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

if [[ ! -d "$CONFIGURED_REPOSITORY" ]]; then
  echo "ViewForge repository not found: $CONFIGURED_REPOSITORY" >&2
  exit 66
fi
REPOSITORY_ROOT="$(cd -- "$CONFIGURED_REPOSITORY" && pwd -P)"
PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing repository virtual environment: $REPOSITORY_ROOT/.venv" >&2
  echo "Create it with the locked uv workflow in docs/VIRTUAL_ENVIRONMENT.md." >&2
  exit 78
fi
if [[ ! -f "$REPOSITORY_ROOT/pyproject.toml" || ! -f "$REPOSITORY_ROOT/src/face3d/local_mcp/server.py" ]]; then
  echo "The selected directory is not a ViewForge 3D source checkout." >&2
  exit 65
fi

PYTHONPATH="$REPOSITORY_ROOT/src" PYTHONNOUSERSITE=1 "$PYTHON" -c '
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("ViewForge 3D Local requires Python 3.11")
import mcp, numpy, PIL, trimesh
import face3d.local_mcp.server
'

mkdir -p "$RUNTIME_HOME/state"
TEMPORARY_LOCATOR="$(mktemp "$RUNTIME_HOME/.repository-root.XXXXXX")"
trap 'rm -f -- "$TEMPORARY_LOCATOR"' EXIT
printf '%s\n' "$REPOSITORY_ROOT" > "$TEMPORARY_LOCATOR"
chmod 600 "$TEMPORARY_LOCATOR"
mv -f -- "$TEMPORARY_LOCATOR" "$RUNTIME_HOME/repository-root"
trap - EXIT

echo "ViewForge 3D Local runtime configured."
echo "Python: $PYTHON"
echo "State: $RUNTIME_HOME/state"
