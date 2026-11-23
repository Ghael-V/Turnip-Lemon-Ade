/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "common/pvr_iface.h"
#include "pdsc/pdsc.h"
#include "pdsc/pdsc_internal.h"
#include "util/macros.h"

#include <stdio.h>
#include <stdint.h>

static inline pdsc_program *pdsc_xfb_query_end_gen(void *mem_ctx,
                                                   enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_query_end");

   ALLOC_TEMP64(p, tzero);
   pdsc_limm(p, pdsc_ref_lo32(tzero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero), pdsc_ref_imm(0));

   ALLOC_TEMP32_ARRAY(p, result, 2);
   pdsc_limm(p, pdsc_ref_elem32(result, 0), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_elem32(result, 1), pdsc_ref_imm(0));

   ALLOC_TEMP32(p, availability);
   pdsc_limm(p, availability, pdsc_ref_imm(1));

   char buf[64];
   pdsc_ref stm_ptemps64[PVR_XFB_MAX_PTEMPS_SETS];
   for (uint32_t i = 0; i < ARRAY_SIZE(stm_ptemps64); i++) {
      snprintf(buf, sizeof(buf), "stm_ptemps64__%u", i * 2);
      stm_ptemps64[i] = pdsc_decl_ptemp64(p, buf, i * 2, true);
   }

   ALLOC_TEMP64(p, prev);
   pdsc_limm(p, pdsc_ref_lo32(prev), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(prev), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, acc);
   ALLOC_TEMP64(p, seg);

   /* Load the counts accumulated so far; skipped when the patch is zero. */
   pdsc_instr_t cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t ld = pdsc_ld(p,
                             prev,
                             PVR_XFB_QUERY_END_REF_ID_QUERY_COUNT_LD_DEST,
                             PVR_XFB_QUERY_END_PATCH_ID_QUERY_COUNT_LD_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, ld, 0));
   pdsc_wdf(p);

   /* Add this segment's counts. The two words are added separately so a carry
    * from one cannot leak into the other.
    */
   const pdsc_ref prev_words[] = { pdsc_ref_lo32(prev), pdsc_ref_hi32(prev) };
   const pdsc_ref seg_words[] = {
      pdsc_ref_lo32(stm_ptemps64[1]),
      pdsc_ref_hi32(stm_ptemps64[1]),
   };

   for (uint32_t i = 0; i < ARRAY_SIZE(prev_words); i++) {
      pdsc_mov32(p, pdsc_ref_lo32(acc), prev_words[i]);
      pdsc_limm(p, pdsc_ref_hi32(acc), pdsc_ref_imm(0));
      pdsc_mov32(p, pdsc_ref_lo32(seg), seg_words[i]);
      pdsc_limm(p, pdsc_ref_hi32(seg), pdsc_ref_imm(0));

      pdsc_add64(p, acc, acc, seg);
      pdsc_mov32(p, pdsc_ref_elem32(result, i), pdsc_ref_lo32(acc));
   }

   /* Store both results to memory; skipped when the patch is zero. */
   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t st = pdsc_st(p,
                             pdsc_ref_elem32(result, 0),
                             PVR_XFB_QUERY_END_REF_ID_RESULT_WRITTEN_ST_SRC0,
                             PVR_XFB_QUERY_END_PATCH_ID_RESULT_WRITTEN_ST_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   /* Update availability */
   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   st = pdsc_st(p,
                availability,
                PVR_XFB_QUERY_END_REF_ID_AVAILABILITY_ST_SRC0,
                PVR_XFB_QUERY_END_PATCH_ID_AVAILABILITY_ST_SRC0,
                false,
                .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   pdsc_halt(p);

   return p;
}
