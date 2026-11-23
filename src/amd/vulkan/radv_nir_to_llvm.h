/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_NIR_TO_LLVM_H
#define RADV_NIR_TO_LLVM_H

#include <stdbool.h>
#include <stdint.h>

#include "amd_family.h"

struct ac_compiler_info;
struct radv_shader_info;
struct nir_shader;
struct radv_shader_binary;
struct radv_shader_args;

struct radv_llvm_compiler_options {
   const struct ac_compiler_info *compiler_info;
   enum radeon_family family;
   uint32_t address32_hi;
   bool robust_buffer_access;
   bool dump_shader;
   bool dump_preoptir;
   bool record_ir;
   bool check_ir;
   bool wgp_mode;
};

void llvm_compile_shader(const struct radv_llvm_compiler_options *options, const struct radv_shader_info *info,
                         unsigned shader_count, struct nir_shader *const *shaders, struct radv_shader_binary **binary,
                         const struct radv_shader_args *args);

#endif /* RADV_NIR_TO_LLVM_H */
