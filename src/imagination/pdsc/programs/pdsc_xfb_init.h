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

#include <stdint.h>
#include <stdio.h>

static inline pdsc_program *pdsc_xfb_init_gen(void *mem_ctx,
                                              enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_init");

   pdsc_ref so_ptemps64[PVR_XFB_MAX_PTEMPS_SETS];
   char buf[64];

   ALLOC_CONST64(p, ptemps, PVR_XFB_INIT_PATCH_ID_PTEMP_LD_COUNT);

   /* Allocate the ptemp buffers */
   for (uint32_t i = 0; i < ARRAY_SIZE(so_ptemps64); i++) {
      snprintf(buf, sizeof(buf), "so_ptemps64_%u", i * 2);
      so_ptemps64[i] = pdsc_decl_ptemp64(p, buf, i * 2, true);
   }

   ALLOC_TEMP64(p, izero);
   pdsc_limm(p, pdsc_ref_lo32(izero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(izero), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, tzero);
   pdsc_limm(p, pdsc_ref_lo32(tzero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, temp_result);

   ALLOC_TEMP32(p, cb_dword0);
   ALLOC_CONST64(p,
                 alignment_sel,
                 PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_DWORD_SELECT);
   ALLOC_TEMP64(p, align_sel);

   ALLOC_CONST64(p,
                 counter_buffers_count,
                 PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_COUNT);

   ALLOC_VAL_CONST64(p, const_one, 1);

   ALLOC_TEMP64_ARRAY(p, counter_buffer, PVR_XFB_MAX_BUFFERS);
   pdsc_ref counter_buffer_array[] = {
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 0)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 0)),
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 1)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 1)),
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 2)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 2)),
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer, 3)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer, 3)),
   };

   for (uint32_t i = 0; i < ARRAY_SIZE(counter_buffer_array); i++)
      pdsc_limm(p, counter_buffer_array[i], pdsc_ref_imm(0));

   ALLOC_TEMP32_ARRAY(p, st_buffer, PVR_XFB_MAX_BUFFERS);
   pdsc_ref st_buffer_array[] = {
      pdsc_ref_elem32(st_buffer, 0),
      pdsc_ref_elem32(st_buffer, 1),
      pdsc_ref_elem32(st_buffer, 2),
      pdsc_ref_elem32(st_buffer, 3),
   };

   for (uint32_t i = 0; i < ARRAY_SIZE(st_buffer_array); i++)
      pdsc_limm(p, pdsc_ref_elem32(st_buffer, i), pdsc_ref_imm(0));

   /* Load ptemps per buffer */
   for (uint32_t i = 0;
        i < ARRAY_SIZE(so_ptemps64) / PVR_XFB_PTEMP_ENTRIES_PER_VARYING;
        i++) {
      pdsc_cmp(p, izero, ptemps, .cop = PDSC_COP_LT);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_ld(p,
              so_ptemps64[i * PVR_XFB_PTEMP_ENTRIES_PER_VARYING],
              PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_0 + i,
              PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_0 + i,
              false,
              .mods.cc = true);

      pdsc_add64(p, izero, izero, const_one, .cc = true);
   }
   pdsc_wdf(p);

   pdsc_instr_t cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);

   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t ld = pdsc_ld(p,
                             counter_buffer,
                             PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_LD_DEST,
                             PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_LD_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, ld, 0));
   pdsc_wdf(p);

   /* Apply counter buffer offsets to the SRC0 ptemp of each buffer group.
    * With fixed stride-3 allocation, group i's SRC0 is always at
    * ptemp64[i * PVR_XFB_PTEMP_ENTRIES_PER_VARYING].
    *
    * The counter buffer value is added to the buffer's SRC0 ptemp (address),
    * advancing the write pointer by the number of bytes already written.
    */
   pdsc_mov64(p, align_sel, alignment_sel);
   pdsc_stmc(p, PDSC_SOMASK_GLOBAL);

   pdsc_limm(p, pdsc_ref_lo32(izero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(izero), pdsc_ref_imm(0));

   for (uint32_t i = 0; i < ARRAY_SIZE(st_buffer_array); i++) {
      /* Group i's SRC0 is at ptemp64[i * PVR_XFB_PTEMP_ENTRIES_PER_VARYING]
       * (fixed stride-3 layout).
       */
      pdsc_ref ptemp64 = so_ptemps64[i * PVR_XFB_PTEMP_ENTRIES_PER_VARYING];

      pdsc_cmp(p, align_sel, pdsc_ref_imm(0), .cop = PDSC_COP_EQ);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_mov32(p, cb_dword0, counter_buffer_array[i]);

      pdsc_cmp(p, align_sel, pdsc_ref_imm(1), .cop = PDSC_COP_EQ);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_mov32(p, cb_dword0, counter_buffer_array[i + 1], .cc = true);

      pdsc_cmp(p, align_sel, pdsc_ref_imm(2), .cop = PDSC_COP_EQ);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_mov32(p, cb_dword0, counter_buffer_array[i + 2], .cc = true);

      pdsc_cmp(p, align_sel, pdsc_ref_imm(3), .cop = PDSC_COP_EQ);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_mov32(p, cb_dword0, counter_buffer_array[i + 3], .cc = true);

      pdsc_cmp(p, izero, counter_buffers_count, .cop = PDSC_COP_LT);
      pdsc_setc(p, PDSC_PRED_P0);

      pdsc_mov32(p, st_buffer_array[i], cb_dword0, .cc = true);

      pdsc_mov32(p, pdsc_ref_lo32(temp_result), cb_dword0, .cc = true);
      pdsc_mov32(p, pdsc_ref_hi32(temp_result), pdsc_ref_imm(0), .cc = true);

      pdsc_add64(p, temp_result, temp_result, ptemp64, .cc = true);
      pdsc_mov64(p, ptemp64, temp_result, .cc = true);

      pdsc_add64(p, izero, izero, const_one, .cc = true);
   }

   cmp = pdsc_cmp(p, tzero, pdsc_ref_null(), .cop = PDSC_COP_NE);

   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_instr_t st = pdsc_st(p,
                             st_buffer_array[0],
                             PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_ST_SRC0,
                             PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_ST_SRC0,
                             false,
                             .mods.cc = true);
   pdsc_src_set(p, cmp, 1, pdsc_src_get(p, st, 0));
   pdsc_wdf(p);

   pdsc_halt(p);

   return p;
}
