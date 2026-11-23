/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PDSC_XFB_MAIN_H
#define PDSC_XFB_MAIN_H

#include "common/pvr_iface.h"
#include "pdsc/pdsc.h"
#include "pdsc/pdsc_internal.h"
#include "pdsc/pdsc_isa_pack.h"
#include "pvr_limits.h"
#include "util/macros.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#define PVR_XFB_SRC3_EOP_BIT_HI32 BITFIELD_BIT(11)

static inline pdsc_program *
pdsc_xfb_main_runtime_gen(void *mem_ctx,
                          enum pdsc_feature features,
                          uint32_t group_count,
                          const uint32_t *group_varying_counts)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_main");
   char buf[64];

   uint32_t total_varyings = 0;
   for (uint32_t g = 0; g < group_count; g++)
      total_varyings += group_varying_counts[g];

   /* Ptemp declarations (stride-3 per group) */
   pdsc_ref
      stm_ptemps64[PVR_XFB_MAX_BUFFERS * PVR_XFB_PTEMP_ENTRIES_PER_VARYING];
   const uint32_t total_ptemps = group_count * PVR_XFB_PTEMP_ENTRIES_PER_VARYING;
   for (uint32_t i = 0; i < total_ptemps; i++) {
      snprintf(buf, sizeof(buf), "stm_ptemps64__%u", i * 2);
      stm_ptemps64[i] = pdsc_decl_ptemp64(p, buf, i * 2, true);
   }

   /* EOP mask: bit 43 of 64-bit SRC3 = bit 11 of hi32 */
   ALLOC_TEMP64(p, eop_mask);
   pdsc_limm(p, pdsc_ref_lo32(eop_mask), pdsc_ref_imm(0));
   pdsc_limm(p,
             pdsc_ref_hi32(eop_mask),
             pdsc_ref_imm(PVR_XFB_SRC3_EOP_BIT_HI32));

   /* Allocate SRC2 constants — flat sequential layout, one per varying */
   assert(total_varyings <= PVR_XFB_MAX_TOTAL_VARYINGS);
   pdsc_ref stm_consts[PVR_XFB_MAX_TOTAL_VARYINGS];
   for (uint32_t i = 0; i < total_varyings; i++) {
      snprintf(buf, sizeof(buf), "stm_src2_%u", i);
      stm_consts[i] =
         pdsc_alloc_const32(p,
                            buf,
                            PDSC_TYPE_IMM,
                            PVR_XFB_MAIN_PATCH_ID_STM_SRC2_CONST_0 + i);
   }

   /* Emit STMs for each group */
   uint32_t varying_idx = 0;
   for (uint32_t g = 0; g < group_count; g++) {
      uint32_t gvc = group_varying_counts[g];
      if (gvc == 0)
         continue;

      uint32_t src0_idx = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING;
      uint32_t src1_idx = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING + 1;
      uint32_t src3_idx = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING + 2;

      /* Non-EOP STMs: varyings 0..(gvc-2) */
      for (uint32_t v = 0; v + 1 < gvc; v++) {
         pdsc_stm(p,
                  stm_ptemps64[src0_idx],
                  stm_ptemps64[src1_idx],
                  stm_consts[varying_idx],
                  stm_ptemps64[src3_idx]);
         varying_idx++;
      }

      /* Last varying: XOR to set EOP, STM, XOR to clear EOP */
      pdsc_sftlp64(p,
                   stm_ptemps64[src3_idx],
                   stm_ptemps64[src3_idx],
                   eop_mask,
                   pdsc_ref_null(),
                   .lop = PDSC_LOP_XOR);
      pdsc_stm(p,
               stm_ptemps64[src0_idx],
               stm_ptemps64[src1_idx],
               stm_consts[varying_idx],
               stm_ptemps64[src3_idx]);
      pdsc_sftlp64(p,
                   stm_ptemps64[src3_idx],
                   stm_ptemps64[src3_idx],
                   eop_mask,
                   pdsc_ref_null(),
                   .lop = PDSC_LOP_XOR);
      varying_idx++;
   }

   pdsc_halt(p);

   pdsc_finalize_program(p);
   pdsc_validate_program(p);
   pdsc_encode_program(p);

   return p;
}

static inline void pdsc_patch_xfb_main(pdsc_program *p,
                                       uint32_t *data,
                                       uint32_t varying_count,
                                       const struct pdsc_stm_src2 *src2_values)
{
   pdsc_patch patches[PVR_XFB_MAX_TOTAL_VARYINGS];
   assert(varying_count <= ARRAY_SIZE(patches));

   for (uint32_t i = 0; i < varying_count; i++) {
      patches[i] = (pdsc_patch){
         .patch_id = PVR_XFB_MAIN_PATCH_ID_STM_SRC2_CONST_0 + i,
         .value = pdsc_stm_src2_pack(p, &src2_values[i]),
      };
   }

   pdsc_patch_program(p, varying_count, patches, data);
}

#undef PVR_XFB_SRC3_EOP_BIT_HI32

#endif /* PDSC_XFB_MAIN_H */
