/*
 * SPDX-License-Identifier: MIT
 */

#include "zink_context.h"
#include "zink_list_restart.h"
#include "zink_resource.h"
#include "zink_screen.h"

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"

#define XXH_INLINE_ALL
#include "util/xxhash.h"

/*
 * Compaction of list topology draws with restart indices
 *
 * The NVIDIA proprietary and NVK Vulkan drivers are slow for list topologies (points, lines, triangles
 * and their adjacency variants) whose indices hit the restart value, while the NVIDIA GL driver
 * rewrites the index buffer. Applications such as blender draw hidden edges as huge GL_LINES draws with restart
 * pairs from a GPU index generator, so the indices cannot be scanned on the CPU. This does the GL
 * driver's rewrite on the GPU: compute passes copy the primitives which hold no restart index, in
 * order, to a scratch index buffer, and the draw becomes an indirect draw of that buffer.
 *
 * Semantics
 *
 * Dropping a primitive which holds a restart index matches the restart semantics only when every
 * run of restart indices starts and ends on a primitive boundary, as the generators write them. A
 * run which does not would also cut the primitives around it, which shows as a mixed primitive:
 * a complete primitive holding both restart and other indices (one of only restart indices is
 * dropped, one of only other indices is kept). The count pass flags a draw with a mixed primitive,
 * and its indices are drawn as they are then.
 *
 * Dropped primitives would also shift the ids of the following ones, so a draw whose bound shaders
 * read the primitive id is left to the original.
 *
 * The draw is an indirect draw of the scratch buffer either way, whose command the scan pass writes.
 * When the count pass flagged the draw, the scan pass gives the command the whole index count and
 * the scatter pass copies the indices as they are, so the draw keeps its restart indices and
 * draws as the original would.
 *
 * Passes (all compute, one dispatch each, with a barrier between them)
 *
 *   count    a subgroup owns a chunk of CHUNK_PRIMS primitives, counts the kept ones, and flags a
 *            mixed primitive in the flag word of the args buffer
 *   scan     one subgroup turns the chunk counts into the first output primitive of each chunk
 *            (an exclusive scan with a carry), writes the draw command, clears the flag, and fills
 *            the stats report
 *   scatter  a subgroup copies the kept primitives of its chunk to the scanned place, in order
 *            (or all indices as they are when the command draws as many as the original)
 *
 * The ordered stream compaction is the usual count, scan, scatter. Within a chunk, the lanes walk
 * their primitives in lockstep rounds and use a subgroup scan for the order, which is why the
 * workaround needs a subgroup size of 32 with the basic and arithmetic operations.
 *
 * Sampling
 *
 * The passes cost something and a draw without restart indices gains nothing from them. The scan
 * pass writes how many indices were dropped to a mapped stats buffer, which the CPU reads after
 * the batch has finished (update_entries). What is learned is kept per draw (bo id, start, count,
 * mode, index size, restart index) in a table with one state per draw:
 *
 *   NEW      nothing is known: the draw runs the passes with a stats slot, which is a sample
 *   PENDING  the sample has not finished: the draw is the original
 *   COMPACT  enough was dropped (MIN_RESTART_INDICES): the draw runs the passes without a stats
 *            slot, and after COMPACT_TTL draws it is sampled again, since the contents of the
 *            indices are not tracked and may have been written again
 *   SKIP     too little was dropped, or the count pass flagged it: the original
 *            draw is used and the draw is never sampled again
 *
 * Only MAX_SAMPLES_IN_FLIGHT samples wait for a report at a time, the other new draws use the
 * original. Draws below MIN_INDICES or above MAX_INDICES never run any of this.
 *
 * Context state
 *
 * The passes are internal work, so the compute shader, its SSBO bindings, and the active queries are
 * saved and restored around them. The per draw parameters come from the stream uploader, and the
 * scratch buffers are kept in the context and grown on demand (and released again when a run of
 * draws needs a quarter or less of them). Because a pass writes the index buffer which the next
 * draw reads, the context's index buffer binding is dropped to force the draw to barrier it.
 */

#define SUBGROUP_SIZE 32
/* every subgroup of a workgroup owns a chunk of primitives, which its invocations walk in lockstep */
#define SUBGROUPS_PER_WORKGROUP 4
#define COPY_WORKGROUP_SIZE (SUBGROUP_SIZE * SUBGROUPS_PER_WORKGROUP)
#define ROUNDS_PER_CHUNK 32
#define CHUNK_PRIMS (SUBGROUP_SIZE * ROUNDS_PER_CHUNK)
#define MAX_INDICES (1u << 26)
#define MIN_INDICES (1 << 19)

#define COMMAND_SIZE sizeof(VkDrawIndexedIndirectCommand)
/* the args buffer holds the draw command and the flag word of the count pass, which is zero between draws */
#define FLAG_OFFSET COMMAND_SIZE
#define ARGS_SIZE (COMMAND_SIZE + 4)
/* the per draw parameters come from the uploader, since buffer writes can be ordered before earlier draws */
#define PARAMS_SIZE 48
#define SSBO_INDICES 0
#define SSBO_OUT 1
#define SSBO_ARGS 2
#define SSBO_PARAMS 3
#define SSBO_COUNTS 4
#define SSBO_STATS 5
#define NUM_SSBOS 6
#define NO_SLOT UINT32_MAX
#define STATS_SLOTS ZINK_LIST_RESTART_STATS_SLOTS
#define STATS_SLOT_SIZE 16
/* a draw of at least this many restart indices pays for the passes */
#define MIN_RESTART_INDICES (1 << 18)
#define COMPACT_TTL 128
/* draws waiting for a report, the others use the original draw */
#define MAX_SAMPLES_IN_FLIGHT 8
/* the scratch indices are released after this many draws which need a quarter or less of them */
#define SHRINK_DRAWS 256
#define SHRINK_MIN_SIZE (4u << 20)
#define MAX_ENTRIES 4096
/* count, scan, scatter */
#define LIST_RESTART_KINDS 3

struct list_params {
   nir_def *start, *count, *restart, *prim_verts;
   nir_def *instances, *base_vertex, *first_instance, *chunks;
   nir_def *slot;
};

/*
 * Loads the per draw parameters from the params SSBO (see the params array in
 * zink_compact_list_restart), which every pass reads the same way.
 */
static struct list_params
load_params(nir_builder *b)
{
   nir_def *p0 = nir_load_ssbo(b, 4, 32, nir_imm_int(b, SSBO_PARAMS), nir_imm_int(b, 0), .align_mul = 16);
   nir_def *p1 = nir_load_ssbo(b, 4, 32, nir_imm_int(b, SSBO_PARAMS), nir_imm_int(b, 16), .align_mul = 16);
   nir_def *p2 = nir_load_ssbo(b, 4, 32, nir_imm_int(b, SSBO_PARAMS), nir_imm_int(b, 32), .align_mul = 16);
   return (struct list_params){
      nir_channel(b, p0, 0), nir_channel(b, p0, 1), nir_channel(b, p0, 2), nir_channel(b, p0, 3),
      nir_channel(b, p1, 0), nir_channel(b, p1, 1), nir_channel(b, p1, 2), nir_channel(b, p1, 3),
      nir_channel(b, p2, 0),
   };
}

/*
 * Starts a compute shader with the given workgroup size and the SSBO count shared by all passes.
 */
static nir_shader *
init_cs_shader(struct zink_screen *screen, const char *name, unsigned workgroup_size)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &screen->nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = workgroup_size;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = NUM_SSBOS;
   return b.shader;
}

/*
 * Turns a finished NIR shader into a compute state of the context.
 */
static void *
finish_cs(struct pipe_context *pctx, nir_shader *nir)
{
   struct pipe_compute_state state = {0};
   state.ir_type = PIPE_SHADER_IR_NIR;
   state.prog = nir;
   return pctx->create_compute_state(pctx, &state);
}

/*
 * Loads the index at element elem of the draw's index buffer, with the size it has in the buffer.
 */
static nir_def *
load_index(nir_builder *b, unsigned index_size, nir_def *elem)
{
   return nir_load_ssbo(b, 1, index_size * 8, nir_imm_int(b, SSBO_INDICES),
                        nir_imul_imm(b, elem, index_size),
                        .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER, .align_mul = index_size);
}

/*
 * Returns whether primitive prim of the draw is copied, which is when it is complete and holds
 * no restart index. When mixed is not NULL it is set to whether the primitive holds both restart
 * and other indices, which makes the draw copy its indices as they are. When indices is not NULL it receives
 * the loaded indices of the primitive, which are only meaningful when it is kept.
 */
static nir_def *
primitive_kept(nir_builder *b, unsigned index_size, unsigned prim_verts, const struct list_params *p,
               nir_def *prim, nir_def **mixed, nir_def **indices)
{
   nir_def *first = nir_imul_imm(b, prim, prim_verts);
   /* a partial last primitive is dropped; the loads of the others stay in range */
   nir_def *valid = nir_uge(b, p->count, nir_iadd_imm(b, first, prim_verts));
   nir_def *base = nir_bcsel(b, valid, nir_iadd(b, p->start, first), nir_imm_int(b, 0));
   nir_def *has_restart = nir_imm_false(b);
   nir_def *all_restart = nir_imm_true(b);
   for (unsigned i = 0; i < prim_verts; i++) {
      nir_def *index = load_index(b, index_size, nir_iadd_imm(b, base, i));
      if (indices)
         indices[i] = index;

      nir_def *is_restart = nir_ieq(b, index_size == 4 ? index : nir_u2u32(b, index), p->restart);
      has_restart = nir_ior(b, has_restart, is_restart);
      all_restart = nir_iand(b, all_restart, is_restart);
   }
   if (mixed)
      *mixed = nir_iand(b, valid, nir_iand(b, has_restart, nir_inot(b, all_restart)));
   return nir_iand(b, valid, nir_inot(b, has_restart));
}

/*
 * Returns whether the count pass flagged the draw, which is when the flag word is set.
 */
static nir_def *
load_flag(nir_builder *b)
{
   return nir_ine_imm(b, nir_load_ssbo(b, 1, 32, nir_imm_int(b, SSBO_ARGS), nir_imm_int(b, FLAG_OFFSET),
                                       .align_mul = 4), 0);
}

/*
 * Returns whether the command draws all the indices of the draw, which is how the scatter pass
 * knows to copy them as they are. The command also does that when nothing was dropped, where the
 * compacted indices are the same as the original ones.
 */
static nir_def *
load_verbatim(nir_builder *b, const struct list_params *p)
{
   return nir_ieq(b, nir_load_ssbo(b, 1, 32, nir_imm_int(b, SSBO_ARGS), nir_imm_int(b, 0), .align_mul = 4), p->count);
}

/*
 * Returns the index of the chunk of primitives which this subgroup owns. The last workgroup may
 * have subgroups whose chunk does not exist.
 */
static nir_def *
subgroup_chunk(nir_builder *b)
{
   nir_def *subgroup = nir_ushr_imm(b, nir_load_local_invocation_index(b), util_logbase2(SUBGROUP_SIZE));
   return nir_iadd(b, nir_imul_imm(b, nir_channel(b, nir_load_workgroup_id(b), 0), SUBGROUPS_PER_WORKGROUP), subgroup);
}

/*
 * Starts the loop over the rounds of a chunk, which the subgroup walks in lockstep, and returns
 * the primitive of this invocation in the round. Ended by pop_round_loop.
 */
static nir_def *
push_round_loop(nir_builder *b, nir_def *chunk, nir_variable *round_var, nir_loop **loop)
{
   nir_store_var(b, round_var, nir_imm_int(b, 0), 0x1);

   *loop = nir_push_loop(b);
   nir_def *round = nir_load_var(b, round_var);
   nir_break_if(b, nir_uge_imm(b, round, ROUNDS_PER_CHUNK));
   return nir_iadd(b, nir_iadd(b, nir_imul_imm(b, chunk, CHUNK_PRIMS), nir_imul_imm(b, round, SUBGROUP_SIZE)),
                   nir_load_subgroup_invocation(b));
}

/*
 * Ends the loop started by push_round_loop and advances to the next round.
 */
static void
pop_round_loop(nir_builder *b, nir_variable *round_var, nir_loop *loop)
{
   nir_store_var(b, round_var, nir_iadd_imm(b, nir_load_var(b, round_var), 1), 0x1);
   nir_pop_loop(b, loop);
}

/*
 * Creates the count pass, which counts the kept primitives of every chunk into the counts buffer
 * and flags a draw with a restart run off a primitive boundary.
 */
static void *
create_count_cs(struct pipe_context *pctx, unsigned index_size, unsigned prim_verts)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   nir_shader *nir = init_cs_shader(screen, "zink_list_restart_count", COPY_WORKGROUP_SIZE);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct list_params p = load_params(&b);
   nir_def *chunk = subgroup_chunk(&b);

   nir_push_if(&b, nir_ult(&b, chunk, p.chunks));
   {
      nir_variable *round_var = nir_local_variable_create(impl, glsl_uint_type(), "round");
      nir_variable *count_var = nir_local_variable_create(impl, glsl_uint_type(), "count");
      nir_store_var(&b, count_var, nir_imm_int(&b, 0), 0x1);

      nir_loop *loop;
      nir_def *prim = push_round_loop(&b, chunk, round_var, &loop);
      {
         nir_def *mixed;
         nir_def *kept = primitive_kept(&b, index_size, prim_verts, &p, prim, &mixed, NULL);
         nir_store_var(&b, count_var, nir_iadd(&b, nir_load_var(&b, count_var), nir_b2i32(&b, kept)), 0x1);
         nir_push_if(&b, mixed);
         {
            nir_store_ssbo(&b, nir_imm_int(&b, 1), nir_imm_int(&b, SSBO_ARGS), nir_imm_int(&b, FLAG_OFFSET),
                           .write_mask = 0x1, .align_mul = 4);
         }
         nir_pop_if(&b, NULL);
      }
      pop_round_loop(&b, round_var, loop);

      nir_def *total = nir_reduce(&b, nir_load_var(&b, count_var), .reduction_op = nir_op_iadd);
      nir_push_if(&b, nir_elect(&b, 1));
      {
         nir_store_ssbo(&b, total, nir_imm_int(&b, SSBO_COUNTS), nir_ishl_imm(&b, chunk, 2),
                        .write_mask = 0x1, .align_mul = 4);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   return finish_cs(pctx, nir);
}

/*
 * Creates the scan pass, which turns the chunk counts into the first kept primitive of each chunk,
 * writes the draw command (all the indices when the draw was flagged, else the kept ones), clears
 * the flag, and writes the report which the CPU samples.
 */
static void *
create_scan_cs(struct pipe_context *pctx)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   nir_shader *nir = init_cs_shader(screen, "zink_list_restart_scan", SUBGROUP_SIZE);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct list_params p = load_params(&b);
   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *counts = nir_imm_int(&b, SSBO_COUNTS);

   nir_variable *round_var = nir_local_variable_create(impl, glsl_uint_type(), "round");
   nir_variable *carry_var = nir_local_variable_create(impl, glsl_uint_type(), "carry");
   nir_store_var(&b, round_var, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, carry_var, nir_imm_int(&b, 0), 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *round = nir_load_var(&b, round_var);
      nir_def *carry = nir_load_var(&b, carry_var);
      /* a lane owns four consecutive chunks per round, which quarters the dependent rounds */
      nir_def *round_base = nir_imul_imm(&b, round, SUBGROUP_SIZE * 4);
      nir_def *chunk = nir_iadd(&b, round_base, nir_imul_imm(&b, lane, 4));
      nir_break_if(&b, nir_uge(&b, round_base, p.chunks));

      nir_def *in_range = nir_ult(&b, chunk, p.chunks);
      nir_def *zero4 = nir_imm_ivec4(&b, 0, 0, 0, 0);
      nir_push_if(&b, in_range);

      nir_def *loaded = nir_load_ssbo(&b, 4, 32, counts, nir_ishl_imm(&b, chunk, 2), .align_mul = 16);
      nir_pop_if(&b, NULL);

      nir_def *value4 = nir_if_phi(&b, loaded, zero4);
      /* the counts buffer is padded to whole vec4s, whose unwritten counts must not be summed */
      nir_def *v[4];
      for (unsigned i = 0; i < 4; i++) {
         v[i] = nir_bcsel(&b, nir_ult(&b, nir_iadd_imm(&b, chunk, i), p.chunks),
                          nir_channel(&b, value4, i), nir_imm_int(&b, 0));
      }

      nir_def *e1 = v[0];
      nir_def *e2 = nir_iadd(&b, e1, v[1]);
      nir_def *e3 = nir_iadd(&b, e2, v[2]);
      nir_def *sum = nir_iadd(&b, e3, v[3]);

      nir_def *before = nir_iadd(&b, carry, nir_exclusive_scan(&b, sum, .reduction_op = nir_op_iadd));
      nir_push_if(&b, in_range);
      nir_store_ssbo(&b, nir_vec4(&b, before, nir_iadd(&b, before, e1), nir_iadd(&b, before, e2), nir_iadd(&b, before, e3)),
                     counts, nir_ishl_imm(&b, chunk, 2), .write_mask = 0xf, .align_mul = 16);
      nir_pop_if(&b, NULL);
      nir_store_var(&b, carry_var, nir_iadd(&b, carry, nir_reduce(&b, sum, .reduction_op = nir_op_iadd)), 0x1);
      nir_store_var(&b, round_var, nir_iadd_imm(&b, round, 1), 0x1);
   }
   nir_pop_loop(&b, loop);

   nir_push_if(&b, nir_elect(&b, 1));
   {
      nir_def *kept = nir_load_var(&b, carry_var);
      nir_def *original = load_flag(&b);
      nir_def *zero = nir_imm_int(&b, 0);
      /* index count, instance count, first index, base vertex, first instance */
      nir_def *cmd[5] = {
         nir_bcsel(&b, original, p.count, nir_imul(&b, kept, p.prim_verts)),
         p.instances, zero, p.base_vertex, p.first_instance,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(cmd); i++) {
         nir_store_ssbo(&b, cmd[i], nir_imm_int(&b, SSBO_ARGS), nir_imm_int(&b, i * 4),
                        .write_mask = 0x1, .align_mul = 4);
      }
      nir_store_ssbo(&b, zero, nir_imm_int(&b, SSBO_ARGS), nir_imm_int(&b, FLAG_OFFSET),
                     .write_mask = 0x1, .align_mul = 4);
      nir_push_if(&b, nir_ine_imm(&b, p.slot, NO_SLOT));
      {
         nir_def *at = nir_imul_imm(&b, p.slot, STATS_SLOT_SIZE);
         nir_store_ssbo(&b, nir_vec2(&b, kept, nir_b2i32(&b, original)), nir_imm_int(&b, SSBO_STATS), at,
                        .write_mask = 0x3, .align_mul = 8);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   return finish_cs(pctx, nir);
}

/*
 * Creates the scatter pass, which copies the kept primitives of every chunk, in order, to the place
 * the scan pass gave them in the scratch index buffer. It copies all the indices as they are when
 * the command draws all of them.
 */
static void *
create_scatter_cs(struct pipe_context *pctx, unsigned index_size, unsigned prim_verts)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   nir_shader *nir = init_cs_shader(screen, "zink_list_restart_scatter", COPY_WORKGROUP_SIZE);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct list_params p = load_params(&b);
   nir_def *verbatim = load_verbatim(&b, &p);
   nir_def *chunk = subgroup_chunk(&b);

   nir_push_if(&b, nir_ult(&b, chunk, p.chunks));
   {
      nir_variable *round_var = nir_local_variable_create(impl, glsl_uint_type(), "round");
      nir_variable *slot_var = nir_local_variable_create(impl, glsl_uint_type(), "slot");
      nir_store_var(&b, slot_var, nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, SSBO_COUNTS), nir_ishl_imm(&b, chunk, 2),
                                                .align_mul = 4), 0x1);

      nir_loop *loop;
      nir_def *prim = push_round_loop(&b, chunk, round_var, &loop);
      nir_push_if(&b, verbatim);
      {
         for (unsigned j = 0; j < prim_verts; j++) {
            nir_def *elem = nir_iadd_imm(&b, nir_imul_imm(&b, prim, prim_verts), j);
            nir_push_if(&b, nir_ult(&b, elem, p.count));
            {
               nir_def *idx = load_index(&b, index_size, nir_iadd(&b, p.start, elem));
               nir_store_ssbo(&b, idx, nir_imm_int(&b, SSBO_OUT), nir_imul_imm(&b, elem, index_size),
                              .write_mask = 0x1, .align_mul = index_size);
            }
            nir_pop_if(&b, NULL);
         }
      }
      nir_push_else(&b, NULL);
      {
         nir_def *indices[6];
         nir_def *kept = primitive_kept(&b, index_size, prim_verts, &p, prim, NULL, indices);
         nir_def *kept_int = nir_b2i32(&b, kept);
         nir_def *slot = nir_load_var(&b, slot_var);
         nir_def *mine = nir_iadd(&b, slot, nir_exclusive_scan(&b, kept_int, .reduction_op = nir_op_iadd));
         nir_push_if(&b, kept);
         {
            for (unsigned j = 0; j < prim_verts; j++) {
               /* the output keeps the size of the indices, which is less to write and to fetch */
               nir_store_ssbo(&b, indices[j], nir_imm_int(&b, SSBO_OUT),
                              nir_imul_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, mine, prim_verts), j), index_size),
                              .write_mask = 0x1, .align_mul = index_size);
            }
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, slot_var, nir_iadd(&b, slot, nir_reduce(&b, kept_int, .reduction_op = nir_op_iadd)), 0x1);
      }
      nir_pop_if(&b, NULL);
      pop_round_loop(&b, round_var, loop);
   }
   nir_pop_if(&b, NULL);
   return finish_cs(pctx, nir);
}

/*
 * Makes sure the scratch buffer *res holds at least size bytes, replacing it with a new one rounded
 * up to a power of two otherwise. Returns false when the allocation fails.
 */
static bool
ensure_buffer(struct pipe_context *pctx, struct pipe_resource **res, unsigned *res_size, unsigned bind, unsigned size)
{
   if (*res_size >= size)
      return true;

   unsigned new_size = util_next_power_of_two(size);
   struct pipe_resource *buf = pipe_buffer_create(pctx->screen, bind, PIPE_USAGE_DEFAULT, new_size);
   if (!buf)
      return false;
   pipe_resource_reference(res, buf);
   pipe_resource_reference(&buf, NULL);
   *res_size = new_size;
   return true;
}

/*
 * Returns the vertices of one primitive for the list topologies which are compacted, else 0.
 */
static unsigned
prim_verts_for_mode(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
   case MESA_PRIM_LINES:
   case MESA_PRIM_TRIANGLES:
   case MESA_PRIM_LINES_ADJACENCY:
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return mesa_vertices_per_prim(mode);
   default:
      return 0;
   }
}

/*
 * Returns the slot holding the count (kind 0) or scatter (kind 1) shader for an index size and
 * primitive size, which are the passes that depend on them.
 */
static void **
cs_slot(struct zink_context *ctx, unsigned kind, unsigned index_size, unsigned prim_verts)
{
   return &ctx->list_restart->cs[kind][index_size >> 1][prim_verts - 1];
}

/* the state of the sampling of a draw, see the design comment at the top */
enum entry_state {
   ENTRY_NEW,
   ENTRY_PENDING,
   ENTRY_COMPACT,
   ENTRY_SKIP,
};

/* what was learned about the indices of a draw */
struct restart_entry {
   enum entry_state state;
   unsigned ttl;
};

/*
 * Returns the key which identifies a draw in the table of what was learned about it. The key uses
 * the id of the memory, which is not reused unlike the address of the resource, so a freed and
 * reallocated index buffer is not mistaken for the one that was sampled.
 */
static uint64_t
draw_key(const struct pipe_draw_info *dinfo, const struct pipe_draw_start_count_bias *draw)
{
   const struct {
      uint32_t bo_id, start, count, restart_index;
      uint8_t mode, index_size;
      uint16_t pad;
   } key = {
      zink_resource(dinfo->index.resource)->obj->bo->unique_id, draw->start, draw->count, dinfo->restart_index,
      dinfo->mode, dinfo->index_size, 0,
   };
   return ((uint64_t)XXH32(&key, sizeof(key), 0) << 32) | XXH32(&key, sizeof(key), 1);
}

/*
 * Forgets everything which was learned about draws, including the draws waiting for a report.
 */
static void
clear_entries(struct zink_context *ctx)
{
   if (ctx->list_restart->entries)
      _mesa_hash_table_u64_destroy(ctx->list_restart->entries);
   ralloc_free(ctx->list_restart->entries_mem);
   ctx->list_restart->entries = NULL;
   ctx->list_restart->entries_mem = NULL;
   ctx->list_restart->num_entries = 0;
   ctx->list_restart->pending_count = 0;
}

/*
 * Returns the entry of a draw key, which is created in the NEW state when the draw is not known.
 * The table is cleared when it is full.
 */
static struct restart_entry *
get_entry(struct zink_context *ctx, uint64_t key)
{
   if (ctx->list_restart->num_entries >= MAX_ENTRIES)
      clear_entries(ctx);
   if (!ctx->list_restart->entries) {
      ctx->list_restart->entries_mem = ralloc_context(NULL);
      ctx->list_restart->entries = _mesa_hash_table_u64_create(ctx->list_restart->entries_mem);
   }

   struct restart_entry *entry = _mesa_hash_table_u64_search(ctx->list_restart->entries, key);
   if (!entry) {
      entry = rzalloc(ctx->list_restart->entries_mem, struct restart_entry);
      _mesa_hash_table_u64_insert(ctx->list_restart->entries, key, entry);
      ctx->list_restart->num_entries++;
   }
   return entry;
}

/*
 * Creates the persistently mapped buffer which the scan pass writes the reports to. Returns false
 * when it cannot be created or mapped.
 */
static bool
ensure_stats(struct zink_context *ctx)
{
   struct pipe_context *pctx = &ctx->base;
   if (ctx->list_restart->stats_map)
      return true;

   struct pipe_resource templ = {0};
   templ.target = PIPE_BUFFER;
   templ.format = PIPE_FORMAT_R8_UNORM;
   templ.bind = PIPE_BIND_SHADER_BUFFER;
   templ.usage = PIPE_USAGE_STAGING;
   templ.flags = PIPE_RESOURCE_FLAG_MAP_PERSISTENT | PIPE_RESOURCE_FLAG_MAP_COHERENT;
   templ.width0 = STATS_SLOTS * STATS_SLOT_SIZE;
   templ.height0 = templ.depth0 = templ.array_size = 1;
   ctx->list_restart->stats = pctx->screen->resource_create(pctx->screen, &templ);
   if (ctx->list_restart->stats) {
      ctx->list_restart->stats_map = pipe_buffer_map(pctx, ctx->list_restart->stats,
                                                    PIPE_MAP_READ | PIPE_MAP_PERSISTENT | PIPE_MAP_COHERENT |
                                                    PIPE_MAP_UNSYNCHRONIZED, &ctx->list_restart->stats_xfer);
   }
   if (!ctx->list_restart->stats_map) {
      pipe_resource_reference(&ctx->list_restart->stats, NULL);
      return false;
   }
   return true;
}

/*
 * Reads the reports of the finished batches, oldest first, and moves the entry of each sampled draw
 * to COMPACT or SKIP.
 */
static void
update_entries(struct zink_context *ctx)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   while (ctx->list_restart->pending_count) {
      const unsigned slot = ctx->list_restart->pending_head;
      const uint32_t *report = ctx->list_restart->stats_map + slot * (STATS_SLOT_SIZE / 4);
      if (!zink_screen_usage_check_completion_fast(screen, ctx->list_restart->pending[slot].usage))
         break;

      struct restart_entry *entry = ctx->list_restart->entries ?
                                    _mesa_hash_table_u64_search(ctx->list_restart->entries,
                                                                ctx->list_restart->pending[slot].key) : NULL;
      if (entry && entry->state == ENTRY_PENDING) {
         /* the draw was left to the original when it was not made of whole primitives */
         const uint64_t dropped = ctx->list_restart->pending[slot].count -
                                  (uint64_t)report[0] * ctx->list_restart->pending[slot].prim_verts;
         if (report[1] || dropped < MIN_RESTART_INDICES) {
            entry->state = ENTRY_SKIP;
         } else {
            entry->state = ENTRY_COMPACT;
            entry->ttl = COMPACT_TTL;
         }
      }
      ctx->list_restart->pending_head = (slot + 1) % STATS_SLOTS;
      ctx->list_restart->pending_count--;
   }
}

/*
 * Returns whether a bound graphics stage reads the primitive id. The primitives dropped by the
 * compaction would shift the ids of the following ones, so such draws are left to the original.
 */
static bool
reads_primitive_id(const struct zink_context *ctx)
{
   for (unsigned i = MESA_SHADER_VERTEX; i <= MESA_SHADER_FRAGMENT; i++) {
      const struct zink_shader *zs = ctx->gfx_stages[i];
      if (zs && (BITSET_TEST(zs->info.system_values_read, SYSTEM_VALUE_PRIMITIVE_ID) ||
                 (zs->info.inputs_read & VARYING_BIT_PRIMITIVE_ID)))
         return true;
   }
   return false;
}

/*
 * Decides whether a draw with restart indices is compacted. When it returns true the passes were
 * launched and the draw is to be replaced by the indirect draw which is written to out_info,
 * out_indirect, and out_draw. When it returns false the original draw is to be used as it is.
 */
bool
zink_compact_list_restart(struct zink_context *ctx,
                          const struct pipe_draw_info *dinfo,
                          const struct pipe_draw_start_count_bias *draw,
                          struct pipe_draw_info *out_info,
                          struct pipe_draw_indirect_info *out_indirect,
                          struct pipe_draw_start_count_bias *out_draw)
{
   struct pipe_context *pctx = &ctx->base;
   struct zink_screen *screen = zink_screen(pctx->screen);
   const unsigned prim_verts = prim_verts_for_mode(dinfo->mode);
   /* the output is kept for the context lifetime */
   if (!prim_verts || draw->count < MIN_INDICES || draw->count > MAX_INDICES ||
       ctx->render_condition_active || reads_primitive_id(ctx))
      return false;
   assert(dinfo->index_size == 1 || dinfo->index_size == 2 || dinfo->index_size == 4);

   if (!ensure_stats(ctx))
      return false;
   update_entries(ctx);

   const uint64_t key = draw_key(dinfo, draw);
   struct restart_entry *entry = get_entry(ctx, key);
   /* sampling a skipped draw again costs more than it could gain, so it stays skipped */
   if (entry->state == ENTRY_PENDING || entry->state == ENTRY_SKIP)
      return false;
   if (entry->state == ENTRY_COMPACT && !--entry->ttl) {
      /* without a free slot the draw keeps compacting and asks again on the next one */
      if (ctx->list_restart->pending_count < MAX_SAMPLES_IN_FLIGHT)
         entry->state = ENTRY_NEW;
      else
         entry->ttl = 1;
   }

   /* only a new entry asks for a report, and only a few draws wait for one at a time */
   unsigned slot = NO_SLOT;
   if (entry->state == ENTRY_NEW) {
      if (ctx->list_restart->pending_count >= MAX_SAMPLES_IN_FLIGHT)
         return false;
      slot = (ctx->list_restart->pending_head + ctx->list_restart->pending_count) % STATS_SLOTS;
   }

   const unsigned chunks = DIV_ROUND_UP(draw->count / prim_verts, CHUNK_PRIMS);
   if (chunks > screen->info.props.limits.maxComputeWorkGroupCount[0])
      return false;

   void **kinds[LIST_RESTART_KINDS] = {
      cs_slot(ctx, 0, dinfo->index_size, prim_verts),
      &ctx->list_restart->scan,
      cs_slot(ctx, 1, dinfo->index_size, prim_verts),
   };
   if (!*kinds[0])
      *kinds[0] = create_count_cs(pctx, dinfo->index_size, prim_verts);
   if (!*kinds[1])
      *kinds[1] = create_scan_cs(pctx);
   if (!*kinds[2])
      *kinds[2] = create_scatter_cs(pctx, dinfo->index_size, prim_verts);
   for (unsigned i = 0; i < LIST_RESTART_KINDS; i++) {
      if (!*kinds[i])
         return false;
   }
   /* one big draw should not keep its scratch indices forever */
   if (ctx->list_restart->out_size >= SHRINK_MIN_SIZE && draw->count * dinfo->index_size <= ctx->list_restart->out_size / 4) {
      if (++ctx->list_restart->small_draws >= SHRINK_DRAWS) {
         pipe_resource_reference(&ctx->list_restart->out, NULL);
         ctx->list_restart->out_size = 0;
         ctx->list_restart->small_draws = 0;
      }
   } else {
      ctx->list_restart->small_draws = 0;
   }

   const bool new_args = !ctx->list_restart->args;
   if (!ensure_buffer(pctx, &ctx->list_restart->args, &ctx->list_restart->args_size,
                      PIPE_BIND_COMMAND_ARGS_BUFFER | PIPE_BIND_SHADER_BUFFER, ARGS_SIZE))
      return false;
   if (new_args) {
      /* the scan pass clears the flag word, so it only has to start out clear */
      const uint32_t clear = 0;
      pctx->buffer_subdata(pctx, ctx->list_restart->args, PIPE_MAP_WRITE, FLAG_OFFSET, sizeof(clear), &clear);
   }
   if (!ensure_buffer(pctx, &ctx->list_restart->out, &ctx->list_restart->out_size,
                      PIPE_BIND_INDEX_BUFFER | PIPE_BIND_SHADER_BUFFER, draw->count * dinfo->index_size) ||
       !ensure_buffer(pctx, &ctx->list_restart->counts, &ctx->list_restart->counts_size,
                      PIPE_BIND_SHADER_BUFFER, align(chunks, 4) * 4))
      return false;

   const uint32_t params[PARAMS_SIZE / 4] = {
      draw->start, draw->count, dinfo->restart_index, prim_verts,
      dinfo->instance_count, draw->index_bias, dinfo->start_instance, chunks,
      slot, 0, 0, 0,
   };
   struct pipe_resource *params_buf = NULL;
   unsigned params_offset;
   u_upload_data_ref(pctx->stream_uploader, 0, sizeof(params), MAX2(16, screen->info.props.limits.minStorageBufferOffsetAlignment), params, &params_offset, &params_buf);
   if (!params_buf)
      return false;

   /* the sets below drop the references of the context */
   struct pipe_shader_buffer saved[NUM_SSBOS] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(saved); i++)
      util_copy_shader_buffer(&saved[i], &ctx->ssbos[MESA_SHADER_COMPUTE][i]);

   const uint32_t saved_writable = ctx->writable_ssbos[MESA_SHADER_COMPUTE] & BITFIELD_MASK(NUM_SSBOS);
   struct zink_compute_program *saved_cs = ctx->curr_compute;

   /* the passes are internal work: pause the queries of the application, they resume with the draws */
   const bool queries_disabled = ctx->queries_disabled;
   if (!queries_disabled)
      pctx->set_active_query_state(pctx, false);

   struct pipe_shader_buffer bufs[NUM_SSBOS] = {
      [SSBO_INDICES] = {dinfo->index.resource, 0, dinfo->index.resource->width0},
      [SSBO_OUT] = {ctx->list_restart->out, 0, ctx->list_restart->out_size},
      [SSBO_ARGS] = {ctx->list_restart->args, 0, ARGS_SIZE},
      [SSBO_PARAMS] = {params_buf, params_offset, sizeof(params)},
      [SSBO_COUNTS] = {ctx->list_restart->counts, 0, ctx->list_restart->counts_size},
      [SSBO_STATS] = {ctx->list_restart->stats, 0, STATS_SLOTS * STATS_SLOT_SIZE},
   };
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, NUM_SSBOS, bufs,
                            BITFIELD_BIT(SSBO_OUT) | BITFIELD_BIT(SSBO_ARGS) | BITFIELD_BIT(SSBO_COUNTS) | BITFIELD_BIT(SSBO_STATS));

   /* count, scan, scatter */
   const unsigned groups[LIST_RESTART_KINDS] = {
      DIV_ROUND_UP(chunks, SUBGROUPS_PER_WORKGROUP), 1, DIV_ROUND_UP(chunks, SUBGROUPS_PER_WORKGROUP),
   };
   const unsigned sizes[LIST_RESTART_KINDS] = {
      COPY_WORKGROUP_SIZE, SUBGROUP_SIZE, COPY_WORKGROUP_SIZE,
   };
   for (unsigned i = 0; i < LIST_RESTART_KINDS; i++) {
      struct pipe_grid_info grid = {0};
      grid.work_dim = 1;
      grid.block[0] = sizes[i];
      grid.block[1] = grid.block[2] = 1;
      grid.grid[0] = groups[i];
      grid.grid[1] = grid.grid[2] = 1;
      /* shader writes are only visible to later work after a barrier, as in GL */
      pctx->memory_barrier(pctx, ctx->memory_barrier | PIPE_BARRIER_SHADER_BUFFER);
      pctx->bind_compute_state(pctx, *kinds[i]);
      pctx->launch_grid(pctx, &grid);
   }
   pctx->memory_barrier(pctx, ctx->memory_barrier | PIPE_BARRIER_INDEX_BUFFER | PIPE_BARRIER_INDIRECT_BUFFER);
   if (slot != NO_SLOT) {
      ctx->list_restart->pending[slot].key = key;
      zink_batch_usage_set(&ctx->list_restart->pending[slot].usage, ctx->bs);
      ctx->list_restart->pending[slot].count = draw->count;
      ctx->list_restart->pending[slot].prim_verts = prim_verts;
      ctx->list_restart->pending_count++;
      entry->state = ENTRY_PENDING;
   }

   pctx->bind_compute_state(pctx, saved_cs);
   if (!queries_disabled)
      pctx->set_active_query_state(pctx, true);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, NUM_SSBOS, saved, saved_writable);
   pipe_resource_reference(&params_buf, NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(saved); i++)
      util_copy_shader_buffer(&saved[i], NULL);

   /* the draw only barriers an index buffer which changed, and the pass wrote this one */
   ctx->index_buffer = NULL;

   *out_info = *dinfo;
   out_info->index_bounds_valid = false;
   out_info->has_user_indices = false;
   out_info->index.resource = ctx->list_restart->out;
   memset(out_indirect, 0, sizeof(*out_indirect));
   out_indirect->buffer = ctx->list_restart->args;
   out_indirect->draw_count = 1;
   out_indirect->stride = COMMAND_SIZE;
   memset(out_draw, 0, sizeof(*out_draw));
   return true;
}

/*
 * Allocates the state of the compaction for a context. Returns false when out of memory.
 */
bool
zink_list_restart_init(struct zink_context *ctx)
{
   ctx->list_restart = CALLOC_STRUCT(zink_list_restart);
   return ctx->list_restart != NULL;
}

/*
 * Frees the state of the compaction of a context, with its shaders, scratch buffers, and entries.
 */
void
zink_list_restart_deinit(struct zink_context *ctx)
{
   struct pipe_context *pctx = &ctx->base;
   if (!ctx->list_restart)
      return;
   void **all = (void **)&ctx->list_restart->cs;
   for (unsigned i = 0; i < sizeof(ctx->list_restart->cs) / sizeof(void *); i++) {
      if (all[i])
         pctx->delete_compute_state(pctx, all[i]);
   }
   if (ctx->list_restart->scan)
      pctx->delete_compute_state(pctx, ctx->list_restart->scan);
   pipe_resource_reference(&ctx->list_restart->out, NULL);
   pipe_resource_reference(&ctx->list_restart->args, NULL);
   pipe_resource_reference(&ctx->list_restart->counts, NULL);
   if (ctx->list_restart->stats_xfer)
      pipe_buffer_unmap(pctx, ctx->list_restart->stats_xfer);
   pipe_resource_reference(&ctx->list_restart->stats, NULL);
   clear_entries(ctx);
   FREE(ctx->list_restart);
   ctx->list_restart = NULL;
}
