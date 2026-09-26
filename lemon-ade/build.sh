#!/bin/sh
# Builds the Lemon-Ade driver zip. Usage: lemon-ade/build.sh [version]
# The version defaults to the current commit; it ends up in lemon_version.h (reported as the
# Vulkan driverInfo) and in the zip's meta.json.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT/build-android"
VERSION=${1:-Lemon-Ade-$(git -C "$ROOT" rev-parse --short HEAD)}

cat > "$ROOT/src/freedreno/vulkan/lemon_version.h" <<EOF
#ifndef LEMON_VERSION_H
#define LEMON_VERSION_H

#define LEMON_ADE_VERSION "$VERSION"

#endif
EOF

if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    meson setup "$BUILD_DIR" "$ROOT" \
        --cross-file="$ROOT/cross-android.txt" \
        -Dbuildtype=release -Dplatforms=android -Dplatform-sdk-version=29 -Dandroid-stub=true \
        -Dgallium-drivers= -Dvulkan-drivers=freedreno -Dfreedreno-kmds=kgsl -Dtools= \
        -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dopengl=false -Dgles1=disabled \
        -Dgles2=disabled -Dshared-glapi=disabled -Db_lto=false -Dzstd=disabled \
        -Dspirv-tools=disabled -Dlibarchive:zstd=disabled -Dlibarchive:xml2=disabled \
        -Dzlib=enabled -Dexpat=disabled
fi
ninja -C "$BUILD_DIR"

ZIP_DIR="$BUILD_DIR/driver_zip"
rm -rf "$ZIP_DIR"
mkdir -p "$ZIP_DIR"
cp "$BUILD_DIR/src/freedreno/vulkan/libvulkan_freedreno.so" "$ZIP_DIR/"
cat > "$ZIP_DIR/meta.json" <<EOF
{
  "schemaVersion": 1,
  "name": "Lemon-Ade Turnip Driver",
  "description": "Adreno 830 Ultra-Optimized Vulkan Driver ($VERSION)",
  "author": "Antigravity and LeYJa",
  "vendor": "Lemon-Ade",
  "driverVersion": "$VERSION",
  "minApi": 30,
  "libraryName": "libvulkan_freedreno.so"
}
EOF
rm -f "$ROOT/Lemon-Ade-Turnip-Driver.zip"
(cd "$ZIP_DIR" && zip -r "$ROOT/Lemon-Ade-Turnip-Driver.zip" .)
echo "Built $VERSION -> $ROOT/Lemon-Ade-Turnip-Driver.zip"
