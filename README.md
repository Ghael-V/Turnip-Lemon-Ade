# Turnip Lemon-Ade

Vulkan driver for Adreno 8xx GPUs, tuned for [Lemon](https://github.com/Ghael-V/Lemon-Project), a Nintendo Switch emulator for Android.

Lemon-Ade is Mesa's Turnip driver on recent Mesa main (since v0.0.8), plus
whitebelyash's generic commits from [`turnip/gen8`](https://github.com/whitebelyash/mesa-unified/tree/turnip/gen8)
(adrenotools versioning, UBWC detection in the fallback gralloc, `disable_gmem`, no concurrent binning). It is
packaged as an [adrenotools](https://github.com/bylaws/libadrenotools) driver zip and loaded by the emulator in
place of the device's stock Qualcomm driver.

## Changes on top of Mesa

- **UBWC swapchains** (since v0.0.9): on A8xx, R8G8B8A8 swapchain and AHB buffers are allocated compressed
  (Qualcomm's private gralloc usage bit). The display hardware only rotates UBWC buffers, so a landscape game on
  a portrait panel no longer makes the compositor redraw the screen on the GPU and preempt the app: in Lemon on
  an Odin 3, 349 GPU preemptions a minute instead of 7,081. The fallback gralloc also reads the UBWC flag from
  snapalloc handles (Snapdragon 8 Elite), which it read as linear before. A one-time probe keeps it off on a
  gralloc it does not recognize; `TU_GRALLOC_UBWC=0` turns it off.
- **Queue priorities on KGSL** (since v0.0.9): `VK_KHR_global_priority` exposes LOW, MEDIUM, HIGH and REALTIME
  and sets the KGSL context priority. REALTIME puts the app on the compositor's ring, where nothing preempts it;
  MEDIUM, the default, is unchanged.
- **Adreno 830 detection:** also recognizes the KGSL revision-0 chip id (`0x44050000`).
- **Float atomics:** `VK_EXT_shader_atomic_float` add operations are exposed and lowered to compare-and-swap loops
  in ir3, for Unreal Engine 5 global illumination / radiance accumulation.
- A8xx advertises fp16 denormal flush-to-zero; the driver reports itself as "Lemon-Ade Turnip Driver" with its
  build version.
- **Faster shader compiles:** ir3's NIR optimization loop gets a single round instead of running to
  convergence (the loop's lowering passes still run to a fixed point afterwards). In Lemon on an Adreno 830
  (Tears of the Kingdom, cold shader cache): per-stage compile time 9.05 -> 6.96 ms, time spent waiting on
  shaders -24%, no extra GPU time in a fixed scene. `IR3_OPT_MAX_ITERS=0` restores upstream behaviour.
- **Global code motion** (`nir_opt_gcm`) is off by default, as upstream: with global value numbering (`GCM=1`,
  the default in v0.0.5 and v0.0.6) it miscompiled Lemon's BCn decoder shader on an Adreno 830.
- Pipeline and KGSL profiling suballocators use 512 KiB blocks instead of 128 KiB.

Lemon-Ade is only tested on the Adreno 830 (Snapdragon 8 Elite).

## Building

Requires the Android NDK r26b at `/opt/android-ndk/android-ndk-r26b` (edit `cross-android.txt` otherwise) and
Mesa's usual build dependencies (meson, ninja, python3 with mako/pyyaml, glslang).

```sh
lemon-ade/build.sh
```

This configures `build-android/`, builds `libvulkan_freedreno.so` and writes `<version>.zip` (the library plus
the adrenotools `meta.json`) at the repository root, e.g. `lemon-ade/build.sh Lemon-Ade-v0.0.9`.

Note: `cross-android.txt` targets `armv8.2-a+dotprod+fp16`, so the library needs a CPU with the dot-product
extension (Snapdragon 845 and later).

## License

Mesa is licensed under the MIT license; see `docs/license.rst`. The original Mesa README is `README.rst`.
