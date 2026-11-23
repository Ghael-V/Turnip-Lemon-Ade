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

static inline pdsc_program *pdsc_xfb_query_begin_gen(void *mem_ctx,
                                                     enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_query_begin");

   ALLOC_TEMP64(p, tzero);
   pdsc_limm(p, pdsc_ref_lo32(tzero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero), pdsc_ref_imm(0));

   ALLOC_CONST32(p,
                 open_segment_scale,
                 PVR_XFB_QUERY_BEGIN_PATCH_ID_OPEN_SEGMENT_SCALE);

   ALLOC_TEMP32_ARRAY(p, result, 2);
   ALLOC_TEMP64(p, scaled);

   ALLOC_TEMP32(p, availability);
   pdsc_limm(p, availability, pdsc_ref_imm(0));

   char buf[64];
   pdsc_ref stm_ptemps64[PVR_XFB_MAX_PTEMPS_SETS];
   for (uint32_t i = 0; i < ARRAY_SIZE(stm_ptemps64); i++) {
      snprintf(buf, sizeof(buf), "stm_ptemps64__%u", i * 2);
      stm_ptemps64[i] = pdsc_decl_ptemp64(p, buf, i * 2, true);
   }

   /* Each transform feedback segment within the query is added to the result
    * slot when it ends. If the query begins inside a segment, start from minus
    * the counts already written in it, so that adding the whole segment leaves
    * only what was written inside the query. The scale is -1 when a segment is
    * open and 0 otherwise, so the result otherwise starts from zero.
    */
   const pdsc_ref seg_words[] = {
      pdsc_ref_lo32(stm_ptemps64[1]),
      pdsc_ref_hi32(stm_ptemps64[1]),
   };

   for (uint32_t i = 0; i < ARRAY_SIZE(seg_words); i++) {
      pdsc_mad(p, scaled, seg_words[i], open_segment_scale, tzero);
      pdsc_mov32(p, pdsc_ref_elem32(result, i), pdsc_ref_lo32(scaled));
   }

   pdsc_instr_t cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t st = pdsc_st(p,
                             pdsc_ref_elem32(result, 0),
                             PVR_XFB_QUERY_BEGIN_REF_ID_RESULT_ST_SRC0,
                             PVR_XFB_QUERY_BEGIN_PATCH_ID_RESULT_ST_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   st = pdsc_st(p,
                availability,
                PVR_XFB_QUERY_BEGIN_REF_ID_AVAILABILITY_ST_SRC0,
                PVR_XFB_QUERY_BEGIN_PATCH_ID_AVAILABILITY_ST_SRC0,
                false,
                .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   pdsc_halt(p);

   return p;
}
