# Copyright © 2017 Broadcom
# SPDX-License-Identifier: MIT

c_args = []
inc_dirs = [
    inc_src, inc_include,
    inc_gallium, inc_gallium_aux, inc_gallium_drivers,
]

if with_panfrost_vdrm
  c_args += '-DHAVE_PANFROST_VDRM'
  inc_dirs += inc_virtio_gpu
endif

libpanfrostwinsys = static_library(
  'panfrostwinsys',
  files('panfrost_drm_winsys.c'),
  include_directories : inc_dirs,
  gnu_symbol_visibility : 'hidden',
  dependencies: [dep_libdrm, idep_mesautil],
  c_args : c_args
)
