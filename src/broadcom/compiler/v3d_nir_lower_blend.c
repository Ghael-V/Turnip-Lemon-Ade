/*
 * Copyright 2025 Raspberry Pi Ltd
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "compiler/nir/nir_lower_blend.h"
#include "v3d_compiler.h"

static bool
lower_dual_color_blend_by_location(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != FRAG_RESULT_DATA1)
      return false;

   sem.location = FRAG_RESULT_DATA0;
   sem.dual_source_blend_index = 1;
   nir_intrinsic_set_io_semantics(intr, sem);
   return true;
}

bool
v3d_nir_lower_blend(nir_shader *nir, struct v3d_compile *c)
{
   if (!c->fs_key->software_blend)
      return false;

   bool progress = false;

   /* Workaround for Unigine Heaven/Valley */
   if (unlikely(c->fs_key->force_dual_color_blend)) {
      nir_variable *var =
         nir_find_variable_with_location(nir, nir_var_shader_out,
                                         FRAG_RESULT_DATA1);
      if (var) {
         var->data.location = FRAG_RESULT_DATA0;
         var->data.index = 1;
      }

      progress = nir_shader_intrinsics_pass(nir,
                                            lower_dual_color_blend_by_location,
                                            nir_metadata_control_flow, NULL);
   }

   nir_lower_blend_options options = {
      /* logic op is handled elsewhere in the compiler */
      .logicop_enable = false,
      .scalar_blend_const = true,
   };

   bool lower_blend = false;
   for (unsigned rt = 0; rt < V3D_MAX_DRAW_BUFFERS; rt++) {
      if (!(c->fs_key->cbufs & (1 << rt))) {
         static const nir_lower_blend_channel replace = {
            .func = PIPE_BLEND_ADD,
            .src_factor = PIPE_BLENDFACTOR_ONE,
            .dst_factor = PIPE_BLENDFACTOR_ZERO,
         };

         options.rt[rt].rgb = replace;
         options.rt[rt].alpha = replace;
         continue;
      }

      lower_blend = true;

      /* Colour write mask is handled by the hardware. */
      options.rt[rt].colormask = 0xf;

      options.rt[rt].format = c->fs_key->color_fmt[rt].format;

      options.rt[rt].rgb.func = c->fs_key->blend[rt].rgb_func;
      options.rt[rt].alpha.func = c->fs_key->blend[rt].alpha_func;
      options.rt[rt].rgb.dst_factor = c->fs_key->blend[rt].rgb_dst_factor;
      options.rt[rt].alpha.dst_factor = c->fs_key->blend[rt].alpha_dst_factor;
      options.rt[rt].rgb.src_factor = c->fs_key->blend[rt].rgb_src_factor;
      options.rt[rt].alpha.src_factor = c->fs_key->blend[rt].alpha_src_factor;
   }

   if (lower_blend)
      progress |= nir_lower_blend(nir, &options);

   return progress;
}
