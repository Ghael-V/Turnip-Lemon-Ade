# Copyright © 2026 Daniel Lang
# SPDX-License-Identifier: MIT

prog_format_table = files('eil_format_table.c.py')
eil_format_table_c = custom_target(
  'eil_format_table.c',
  input: files(
    'eil_format_table.yaml'
  ),
  output: 'eil_format_table.c',
  command: [prog_python, prog_format_table, '--yaml', '@INPUT@', '--out', '@OUTPUT@'],
)

layout_files = [
  'eil_format.c',
  'eil_format.h',
  eil_format_table_c,
]

libetnaviv_layout = static_library(
  'libetnaviv_layout',
  layout_files,
  include_directories: [
    inc_etnaviv,
  ],
  dependencies: idep_mesautil,
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'etnaviv_layout_tests',
    executable(
      'etnaviv_layout_tests',
      files(
        'tests/format-test.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories: [
        inc_etnaviv,
      ],
      dependencies: [idep_gtest, idep_mesautil],
      link_with : [libetnaviv_layout],
    ),
    suite : ['etnaviv'],
    protocol : 'gtest',
  )
endif
