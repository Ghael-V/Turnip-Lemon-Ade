/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "common/pvr_iface.h"
#include "pdsc/pdsc.h"
#include "pdsc/pdsc_internal.h"
#include "pvr_limits.h"
#include "util/macros.h"

#include <stdio.h>
#include <stdint.h>

static inline pdsc_program *pdsc_xfb_end_gen(void *mem_ctx,
                                             enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_end");

   ALLOC_CONST64(p, counter_count, PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_COUNT);
   ALLOC_CONST32(p,
                 counter_buffer_stride_0,
                 PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_0);
   ALLOC_CONST32(p,
                 counter_buffer_stride_1,
                 PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_1);
   ALLOC_CONST32(p,
                 counter_buffer_stride_2,
                 PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_2);
   ALLOC_CONST32(p,
                 counter_buffer_stride_3,
                 PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_3);
   ALLOC_VAL_CONST64(p, const_one, 1);

   pdsc_ref counter_buffer_stride_arr[] = {
      counter_buffer_stride_0,
      counter_buffer_stride_1,
      counter_buffer_stride_2,
      counter_buffer_stride_3,
   };

   ALLOC_TEMP64(p, izero);
   pdsc_limm(p, pdsc_ref_lo32(izero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(izero), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, tzero);
   pdsc_limm(p, pdsc_ref_lo32(tzero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero), pdsc_ref_imm(0));

   ALLOC_TEMP64_ARRAY(p, counter_buffer, PVR_XFB_MAX_BUFFERS / 2);
   pdsc_ref counter_buffer_array[] = {
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 0)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 0)),
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 1)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 1)),
   };

   ALLOC_TEMP32_ARRAY(p, st_buffer, PVR_XFB_MAX_BUFFERS);
   pdsc_ref st_buffer_array[] = {
      pdsc_ref_elem32(st_buffer, 0),
      pdsc_ref_elem32(st_buffer, 1),
      pdsc_ref_elem32(st_buffer, 2),
      pdsc_ref_elem32(st_buffer, 3),
   };

   ALLOC_TEMP64_ARRAY(p, mad_result, PVR_XFB_MAX_BUFFERS);

   char buf[64];
   pdsc_ref stm_ptemps64[PVR_XFB_MAX_PTEMPS_SETS];
   for (uint32_t i = 0; i < ARRAY_SIZE(stm_ptemps64); i++) {
      snprintf(buf, sizeof(buf), "stm_ptemps64__%u", i * 2);
      stm_ptemps64[i] = pdsc_decl_ptemp64(p, buf, i * 2, true);
   }

   /* Load counter buffers */
   pdsc_instr_t cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);

   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t ld = pdsc_ld(p,
                             counter_buffer,
                             PVR_XFB_END_REF_ID_COUNTER_BUFFER_LD_DEST,
                             PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_LD_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, ld, 0));
   pdsc_wdf(p);

   /* Update the number of primitives processed from the ptemps. */
   for (uint32_t i = 0; i < ARRAY_SIZE(st_buffer_array); i++) {
      pdsc_cmp(p, izero, counter_count, .cop = PDSC_COP_LT);
      pdsc_setc(p, PDSC_PRED_P0);

      pdsc_ref mad_dst = pdsc_ref_elem64(mad_result, i);
      pdsc_mov32(p, pdsc_ref_lo32(mad_dst), counter_buffer_array[i], .cc = true);
      pdsc_mov32(p, pdsc_ref_hi32(mad_dst), pdsc_ref_imm(0), .cc = true);

      pdsc_mad(
         p,
         mad_dst,
         pdsc_ref_lo32(stm_ptemps64[i * PVR_XFB_PTEMP_ENTRIES_PER_VARYING + 1]),
         counter_buffer_stride_arr[i],
         mad_dst,
         .cc = true);

      pdsc_mov32(p, st_buffer_array[i], pdsc_ref_lo32(mad_dst), .cc = true);
      pdsc_add64(p, izero, izero, const_one, .cc = true);
   }

   /* Store counter buffer after updating with number of primitives processed */

   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);

   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t st = pdsc_st(p,
                             st_buffer_array[0],
                             PVR_XFB_END_REF_ID_COUNTER_BUFFER_ST_SRC0,
                             PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_ST_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   /* Accumulate this segment's primitive counts into the active stream query,
    * as the next transform feedback begin resets them. The load and store are
    * skipped when their patches are zero, i.e. when no query is active.
    *
    * A plain load/add/store is sufficient: on Rogue this program runs on a
    * single PDS, so there is no concurrent update of the result slot.
    */
   ALLOC_TEMP64(p, query_prev);
   pdsc_limm(p, pdsc_ref_lo32(query_prev), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(query_prev), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, query_acc);
   ALLOC_TEMP64(p, query_seg);
   ALLOC_TEMP32_ARRAY(p, query_result, 2);

   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   ld = pdsc_ld(p,
                query_prev,
                PVR_XFB_END_REF_ID_QUERY_RESULT_LD_DEST,
                PVR_XFB_END_PATCH_ID_QUERY_RESULT_LD_SRC0,
                false,
                .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, ld, 0));
   pdsc_wdf(p);

   /* The two words are added separately so a carry from one cannot leak into
    * the other.
    */
   const pdsc_ref query_prev_words[] = {
      pdsc_ref_lo32(query_prev),
      pdsc_ref_hi32(query_prev),
   };
   const pdsc_ref query_seg_words[] = {
      pdsc_ref_lo32(stm_ptemps64[1]),
      pdsc_ref_hi32(stm_ptemps64[1]),
   };

   for (uint32_t i = 0; i < ARRAY_SIZE(query_prev_words); i++) {
      pdsc_mov32(p, pdsc_ref_lo32(query_acc), query_prev_words[i]);
      pdsc_limm(p, pdsc_ref_hi32(query_acc), pdsc_ref_imm(0));
      pdsc_mov32(p, pdsc_ref_lo32(query_seg), query_seg_words[i]);
      pdsc_limm(p, pdsc_ref_hi32(query_seg), pdsc_ref_imm(0));

      pdsc_add64(p, query_acc, query_acc, query_seg);
      pdsc_mov32(p, pdsc_ref_elem32(query_result, i), pdsc_ref_lo32(query_acc));
   }

   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   st = pdsc_st(p,
                pdsc_ref_elem32(query_result, 0),
                PVR_XFB_END_REF_ID_QUERY_RESULT_ST_SRC0,
                PVR_XFB_END_PATCH_ID_QUERY_RESULT_ST_SRC0,
                false,
                .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   pdsc_halt(p);

   return p;
}
