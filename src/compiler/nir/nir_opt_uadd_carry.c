/*
 * Copyright © 2026 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* This pass creates:
 * - uadd_with_carry from (a + b) < a
 * - usub_with_borrow from a < b, a - b
 *
 * The comparisons are open coded or lowered carry/borrow expressions.
 *
 * This optimization can't be done with algebraic patterns because
 * we need to replace both the uses of the addition and of the comparison.
 * In the subtract/borrow case, there is even a direct dependency between
 * the matched ops.
 *
 * The usub_with_borrow optimization requires the pass to be run after
 * nir_opt_algebraic_late with has_isub.
 */

#include "nir.h"
#include "nir_builder.h"

#include "util/set.h"

#define XXH_INLINE_ALL
#include "util/xxhash.h"

#define HASH(hash, data) XXH32(&(data), sizeof(data), hash)

static uint32_t
hash_borrow_instr(const void *data)
{
   const nir_alu_instr *alu = data;

   uint32_t hash = 0;

   for (unsigned i = 0; i < 2; i++) {
      hash = HASH(hash, alu->src[0].swizzle[0]);
      hash = HASH(hash, alu->src[0].src.ssa);
   }

   return hash;
}

static bool
cmp_borrow_func(const void *data0, const void *data1)
{
   const nir_alu_instr *alu0 = data0;
   const nir_alu_instr *alu1 = data1;

   return nir_alu_srcs_equal(alu0, alu1, 0, 0) &&
          nir_alu_srcs_equal(alu0, alu1, 1, 1);
}

static bool
borrow_set_add_or_opt(nir_builder *b, nir_alu_instr *alu, struct set *sub_borrow_set)
{
   struct set_entry *e = _mesa_set_search_or_add(sub_borrow_set, alu, NULL);
   nir_alu_instr *match = (nir_alu_instr *)e->key;
   if (match == alu)
      return false;

   if ((match->op == nir_op_isub) != (alu->op == nir_op_isub) &&
       nir_block_dominates(match->instr.block, alu->instr.block)) {
      /* If one instruction is a sub (and one is a compare),
       * we can combine them to usub_with_borrow_out.
       */
      nir_alu_instr *isub = match->op == nir_op_isub ? match : alu;
      nir_alu_instr *cmp = match->op == nir_op_isub ? alu : match;

      b->cursor = nir_before_instr(&match->instr);

      nir_def *usub_with_borrow = nir_usub_with_borrow_out(b, nir_mov_alu(b, isub->src[0], 1),
                                                           nir_mov_alu(b, isub->src[1], 1));
      nir_def_replace(&isub->def, nir_channel(b, usub_with_borrow, 0));

      nir_def *borrow = nir_i2b(b, nir_channel(b, usub_with_borrow, 1));

      if (cmp->op == nir_op_uge)
         borrow = nir_inot(b, borrow);

      nir_def_replace(&cmp->def, borrow);

      _mesa_set_remove(sub_borrow_set, e);
      return true;
   } else {
      /* otherwise, replace hashed instruction */
      e->key = alu;
      return false;
   }
}

static bool
opt_uadd_carry_cmp(nir_builder *b, nir_alu_instr *alu)
{
   nir_scalar cmp_src[2] = {
      nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]),
      nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]),
   };

   /* Check if this is ult(iadd(a, b), a). */
   if (!nir_scalar_is_alu(cmp_src[0]))
      return false;

   nir_alu_instr *iadd = nir_scalar_as_alu(cmp_src[0]);
   if (iadd->op != nir_op_iadd || iadd->def.num_components != 1)
      return false;

   if (!nir_scalar_equal(nir_scalar_chase_alu_src(cmp_src[0], 0), cmp_src[1]) &&
       !nir_scalar_equal(nir_scalar_chase_alu_src(cmp_src[0], 1), cmp_src[1]))
      return false;

   b->cursor = nir_before_instr(&iadd->instr);

   /* Replace the iadd with uadd_with_carry_out. */
   nir_def *add_with_carry = nir_uadd_with_carry_out(b, nir_mov_alu(b, iadd->src[0], 1),
                                                     nir_mov_alu(b, iadd->src[1], 1));
   nir_def_replace(&iadd->def, nir_channel(b, add_with_carry, 0));

   nir_def *carry = nir_i2b(b, nir_channel(b, add_with_carry, 1));

   if (alu->op == nir_op_uge)
      carry = nir_inot(b, carry);

   nir_def_replace(&alu->def, carry);
   return true;
}

static bool
opt_uadd_carry_alu(nir_builder *b, nir_alu_instr *alu,
                   struct set *borrow_set, unsigned bit_sizes)
{
   if (alu->def.num_components != 1)
      return false;

   switch (alu->op) {
   case nir_op_ult:
   case nir_op_uge:
      if (!(alu->src[0].src.ssa->bit_size & bit_sizes))
         return false;
      if (opt_uadd_carry_cmp(b, alu))
         return true;

      return borrow_set_add_or_opt(b, alu, borrow_set);

   case nir_op_isub:
      if (!(alu->def.bit_size & bit_sizes))
         return false;
      return borrow_set_add_or_opt(b, alu, borrow_set);

   case nir_op_b2i8:
   case nir_op_b2i16:
   case nir_op_b2i32:
   case nir_op_b2i64: {
      /* Because NIR forces us to output an integer vector instead of an integer and a boolean,
       * we might end up with b2i(ine(uadd_with_carry_out(a, b).y, 0)) when the carry is actually
       * used as an integer. Copy propagate this to just uadd_with_carry_out(a, b).y
       */
      nir_scalar src = nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
      if (!nir_scalar_is_alu(src) || nir_scalar_alu_op(src) != nir_op_ine)
         return false;

      nir_scalar cmp_src[2] = {
         nir_scalar_chase_alu_src(src, 0),
         nir_scalar_chase_alu_src(src, 1),
      };

      for (unsigned i = 0; i < 2; i++) {
         if (!nir_scalar_is_const(cmp_src[!i]) || nir_scalar_as_uint(cmp_src[!i]) != 0)
            continue;
         if (cmp_src[i].comp != 1 || cmp_src[i].def-> bit_size != alu->def.bit_size ||
             !nir_scalar_is_alu(cmp_src[i]))
            continue;

         switch (nir_scalar_alu_op(cmp_src[i])) {
         case nir_op_uadd_with_carry_out:
         case nir_op_usub_with_borrow_out:
            break;
         default:
            continue;
         }

         b->cursor = nir_before_instr(&alu->instr);

         nir_def_replace(&alu->def, nir_mov_scalar(b, cmp_src[i]));
         return true;
      }
      return false;
   }
   default:
      return false;
   }
}

static bool
opt_uadd_carry_impl(nir_function_impl *impl, unsigned bit_sizes)
{
   nir_metadata_require(impl, nir_metadata_dominance);

   struct set borrow_set;
   _mesa_set_init(&borrow_set, NULL, hash_borrow_instr, cmp_borrow_func);

   nir_builder b = nir_builder_create(impl);

   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_alu)
            progress |= opt_uadd_carry_alu(&b, nir_instr_as_alu(instr),
                                           &borrow_set, bit_sizes);
      }
   }

   _mesa_set_fini(&borrow_set, NULL);

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
nir_opt_uadd_carry(nir_shader *shader, unsigned bit_sizes)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      progress |= opt_uadd_carry_impl(impl, bit_sizes);
   }

   return progress;
}
