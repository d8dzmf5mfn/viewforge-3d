#!/bin/zsh
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd)"
LOCAL_APP_ROOT="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "$LOCAL_APP_ROOT/.." && pwd)"
CONFIGURATION="${1:-debug}"
OUTPUT_ROOT="$REPOSITORY_ROOT/dist/viewforge-local"
OUTPUT_APP="$OUTPUT_ROOT/ViewForge Local.app"
STAGING_ROOT="$(mktemp -d /tmp/viewforge-local-app.XXXXXX)"
STAGING_APP="$STAGING_ROOT/ViewForge Local.app"
CONTENTS="$STAGING_APP/Contents"
RESOURCES="$CONTENTS/Resources"

cleanup() {
  rm -rf -- "$STAGING_ROOT"
}
trap cleanup EXIT

swift build --package-path "$LOCAL_APP_ROOT" --configuration "$CONFIGURATION"
BIN_DIRECTORY="$(swift build --package-path "$LOCAL_APP_ROOT" --configuration "$CONFIGURATION" --show-bin-path)"

mkdir -p "$CONTENTS/MacOS" "$RESOURCES/python"
cp "$BIN_DIRECTORY/ViewForgeLocal" "$CONTENTS/MacOS/ViewForgeLocal"
cp "$LOCAL_APP_ROOT/Resources/Info.plist" "$CONTENTS/Info.plist"

PYTHON_EXECUTABLE="$(uv python find --managed-python --no-project --directory /tmp 3.11)"
PYTHON_PREFIX="$($PYTHON_EXECUTABLE -c 'import sys; print(sys.prefix)')"
rsync -a "$PYTHON_PREFIX/" "$RESOURCES/runtime/"
LOCKED_REQUIREMENTS="$STAGING_ROOT/viewforge-runtime-requirements.txt"
uv export \
  --quiet \
  --project "$REPOSITORY_ROOT" \
  --frozen \
  --no-dev \
  --no-emit-project \
  --format requirements.txt \
  --output-file "$LOCKED_REQUIREMENTS"
uv pip install \
  --python "$RESOURCES/runtime/bin/python3.11" \
  --break-system-packages \
  --requirements "$LOCKED_REQUIREMENTS"

BUNDLED_PYTHON_LITERAL="'$RESOURCES/runtime/bin/python3.11'"
BUNDLED_PYTHON_REPLACEMENT='"$(dirname "$0")/python3.11"'
for launcher in "$RESOURCES/runtime/bin/"*; do
  if [[ -f "$launcher" ]] && [[ "$(head -n 1 "$launcher")" == '#!/bin/sh' ]]; then
    BUNDLED_PYTHON_LITERAL="$BUNDLED_PYTHON_LITERAL" \
      BUNDLED_PYTHON_REPLACEMENT="$BUNDLED_PYTHON_REPLACEMENT" \
      perl -pi -e \
        's{\Q$ENV{BUNDLED_PYTHON_LITERAL}\E}{$ENV{BUNDLED_PYTHON_REPLACEMENT}}g' \
        "$launcher"
  fi
done

PYTHON_PREFIX="$PYTHON_PREFIX" perl -pi -e \
  's|\Q$ENV{PYTHON_PREFIX}\E|/opt/viewforge-python|g' \
  "$RESOURCES/runtime/lib/python3.11/_sysconfigdata__darwin_darwin.py"
install_name_tool \
  -id "@rpath/libpython3.11.dylib" \
  "$RESOURCES/runtime/lib/libpython3.11.dylib"

TOOLCHAIN_RPATHS=("${(@f)$(
  otool -l "$CONTENTS/MacOS/ViewForgeLocal" |
    awk '/LC_RPATH/{reading=1; next} reading && /path /{print $2; reading=0}' |
    rg '^/Users/|^/Volumes/' || true
)}")
for rpath in "${TOOLCHAIN_RPATHS[@]}"; do
  if [[ -n "$rpath" ]]; then
    install_name_tool -delete_rpath "$rpath" "$CONTENTS/MacOS/ViewForgeLocal"
  fi
done
strip -S -x "$CONTENTS/MacOS/ViewForgeLocal"

rsync -a \
  --exclude '__pycache__' \
  "$REPOSITORY_ROOT/src/face3d/" \
  "$RESOURCES/python/face3d/"
rsync -a \
  --exclude '__pycache__' \
  "$REPOSITORY_ROOT/plugins/viewforge-3d-toolkit/" \
  "$RESOURCES/viewforge-3d-toolkit/"

# Managed Python distributions and wheel installation can carry bytecode whose code objects retain
# build-machine source paths. The app disables bytecode writes at runtime, so remove every bundled
# cache before signing to keep the bundle free of workstation provenance.
find "$RESOURCES" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "$RESOURCES" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

codesign --force --deep --sign - "$STAGING_APP"
mkdir -p "$OUTPUT_ROOT"
if [[ -e "$OUTPUT_APP" ]]; then
  BACKUP_APP="$OUTPUT_ROOT/ViewForge Local.previous-$(date -u +%Y%m%dT%H%M%SZ).app"
  mv "$OUTPUT_APP" "$BACKUP_APP"
fi
mv "$STAGING_APP" "$OUTPUT_APP"

echo "$OUTPUT_APP"
