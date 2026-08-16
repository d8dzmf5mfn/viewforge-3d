#!/bin/zsh
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../.." && pwd)"
SOURCE_APP="${1:-$REPOSITORY_ROOT/dist/viewforge-local/ViewForge Local.app}"
INSTALL_ROOT="${VIEWFORGE_INSTALL_ROOT:-${HOME:?}/Applications}"
TARGET_APP="$INSTALL_ROOT/ViewForge Local.app"

if [[ ! -x "$SOURCE_APP/Contents/MacOS/ViewForgeLocal" ]]; then
  echo "ViewForge Local executable not found at: $SOURCE_APP" >&2
  exit 66
fi

if [[ ! -x "$SOURCE_APP/Contents/Resources/runtime/bin/python3.11" ]] || \
   [[ ! -d "$SOURCE_APP/Contents/Resources/python/face3d" ]]; then
  echo "ViewForge Local bundled Python runtime is incomplete: $SOURCE_APP" >&2
  exit 66
fi

mkdir -p "$INSTALL_ROOT"
STAGING_ROOT="$(mktemp -d "$INSTALL_ROOT/.viewforge-local-install.XXXXXX")"
STAGING_APP="$STAGING_ROOT/ViewForge Local.app"

cleanup() {
  rm -rf -- "$STAGING_ROOT"
}
trap cleanup EXIT

rsync -a "$SOURCE_APP/" "$STAGING_APP/"
codesign --verify --deep --strict "$STAGING_APP"

if [[ -e "$TARGET_APP" ]]; then
  BACKUP_APP="$INSTALL_ROOT/ViewForge Local.previous-$(date -u +%Y%m%dT%H%M%SZ).app"
  if [[ -e "$BACKUP_APP" ]]; then
    echo "Backup path already exists: $BACKUP_APP" >&2
    exit 73
  fi
  mv "$TARGET_APP" "$BACKUP_APP"
  echo "Previous app preserved at: $BACKUP_APP"
fi

mv "$STAGING_APP" "$TARGET_APP"
echo "$TARGET_APP"
