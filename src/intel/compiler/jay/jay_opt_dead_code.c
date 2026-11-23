/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "jay_builder.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

static void
pass(jay_function *f)
{
   bool dead_phis;
   do {
      BITSET_WORD *live_set = BITSET_CALLOC(f->ssa_alloc);
      dead_phis = false;

      jay_foreach_block_rev(f, block) {
         jay_foreach_inst_in_block_safe_rev(block, I) {
            bool keep = jay_opcode_infos[I->op].side_effects;
            unsigned dst = jay_base_index(I->dst);
            if (!BITSET_TEST_COUNT(live_set, dst, jay_num_values(I->dst)) &&
                I->op != JAY_OPCODE_SEND) {

               /* Instructions like "mov.f32.ne.f0.0 _, r0.f64" are invalid due
                * to stride restrictions, so leave the strided destination in
                * this case. This could be relaxed.
                */
               if (I->type == jay_src_type(I, 0)) {
                  if (I->predication == 2 && !jay_is_null(I->dst)) {
                     jay_shrink_sources(I, I->num_srcs - 1);
                     I->predication = 1;
                  }

                  I->dst = jay_null();
               }
            } else {
               keep = true;
            }

            if (!jay_is_null(I->cond_flag) &&
                !BITSET_TEST(live_set, jay_index(I->cond_flag)) &&
                (I->op != JAY_OPCODE_CMP || jay_is_null(I->dst)) &&
                I->op != JAY_OPCODE_SEND) {

               I->cond_flag = jay_null();
               I->conditional_mod = GEN_CONDITION_NONE;
               I->zero_inactive = false;
            } else if (!jay_is_null(I->cond_flag)) {
               keep = true;
            }

            if (!keep) {
               jay_remove_instruction(I);
               dead_phis |= (I->op == JAY_OPCODE_PHI_DST);
            } else {
               jay_foreach_src_index(I, s, _, index) {
                  BITSET_SET(live_set, index);
               }
            }
         }

         if (dead_phis) {
            jay_foreach_predecessor(block, pred, UGPR) {
               jay_foreach_phi_src_in_block(*pred, phi_src) {
                  if (!BITSET_TEST(live_set, jay_phi_src_index(phi_src))) {
                     jay_remove_instruction(phi_src);
                  }
               }
            }
         }
      }

      free(live_set);
   } while (dead_phis);
}

JAY_DEFINE_FUNCTION_PASS(jay_opt_dead_code, pass)
