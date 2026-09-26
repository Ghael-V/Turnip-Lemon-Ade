# Turnip Lemon-Ade

Vulkan driver for Adreno 8xx GPUs, tuned for [Lemon](https://github.com/Ghael-V/Lemon-Project), a Nintendo Switch emulator for Android.

Lemon-Ade is Mesa's Turnip driver (branch `lemon-ade`) on top of whitebelyash's
[`turnip/gen8`](https://github.com/whitebelyash/mesa-unified/tree/turnip/gen8) branch, which adds
Adreno 8xx (A810–A840) support to upstream Mesa. It is packaged as an
[adrenotools](https://github.com/bylaws/libadrenotools) driver zip and loaded by the emulator in place of the
device's stock Qualcomm driver.

## Changes on top of `turnip/gen8`

- **Adreno 830 detection:** also recognizes the KGSL revision-0 chip id (`0x44050000`) in the device table and the
  ir3 GPU profile, and fixes `is_a830` in `tu_pipeline.cc`, which was always true
  (`chip_id == X || 0x44050001`), so every GPU took the A8xx-only pipeline path.
- **KGSL:** fixed a leaked fence fd when a sparse-bind submit fails.
- **Float atomics:** `VK_EXT_shader_atomic_float` add operations are exposed and lowered to compare-and-swap loops
  in ir3, for Unreal Engine 5 global illumination / radiance accumulation.
- A8xx advertises fp16 denormal flush-to-zero; the driver reports itself as "Lemon-Ade Turnip Driver" with its
  build version.

These apply to every Adreno generation the build supports, but Lemon-Ade is only tested on the Adreno 830
(Snapdragon 8 Elite).

## Building

Requires the Android NDK r26b at `/opt/android-ndk/android-ndk-r26b` (edit `cross-android.txt` otherwise) and
Mesa's usual build dependencies (meson, ninja, python3 with mako/pyyaml, glslang).

```sh
lemon-ade/build.sh
```

This configures `build-android/`, builds `libvulkan_freedreno.so` and writes `Lemon-Ade-Turnip-Driver.zip` (the
library plus the adrenotools `meta.json`) at the repository root.

Note: `cross-android.txt` targets `armv8.2-a+dotprod+fp16`, so the library needs a CPU with the dot-product
extension (Snapdragon 845 and later).

## License

Mesa is licensed under the MIT license; see `docs/license.rst`. The original Mesa README is `README.rst`.
