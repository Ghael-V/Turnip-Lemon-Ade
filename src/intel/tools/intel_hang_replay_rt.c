/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "intel_tools.h"
#include "intel_hang_replay_lib.h"
#include "intel_hang_replay_rt.h"

struct replay_validation {
   const uint8_t *file_map;
   size_t file_size;
   struct util_dynarray *buffers;

   bool unresolved;
   uint64_t reported[64];
   unsigned reported_count;
   unsigned unresolved_count;
};

static uint64_t
replay_u64_from_dwords(const uint32_t *dw)
{
   return dw[0] | (uint64_t)dw[1] << 32;
}

static bool
replay_reference_reported(struct replay_validation *validation,
                          uint64_t address)
{
   for (unsigned i = 0; i < validation->reported_count; i++) {
      if (validation->reported[i] == address)
         return true;
   }

   if (validation->reported_count < ARRAY_SIZE(validation->reported))
      validation->reported[validation->reported_count++] = address;

   return false;
}

static void
replay_validate_gpu_reference(struct replay_validation *validation,
                              const char *name, uint64_t address)
{
   if (!address || replay_reference_reported(validation, address))
      return;

   enum intel_hang_replay_address_status status =
      intel_hang_replay_address_status(validation->buffers, address);

   if (status == INTEL_HANG_REPLAY_ADDRESS_CAPTURED)
      return;

   if (!validation->unresolved) {
      fprintf(stderr,
              "\nWARNING: Replay input contains unresolved GPU references\n\n");
      validation->unresolved = true;
   }

   validation->unresolved_count++;

   if (validation->unresolved_count <= 5) {
      fprintf(stderr, "  %-20s 0x%016"PRIx64"  [%s]\n",
              name, address,
              status == INTEL_HANG_REPLAY_ADDRESS_MAPPED ?
              "CONTENTS NOT CAPTURED" : "VMA NOT CAPTURED");
   }
}

static bool
replay_read_captured_gpu_data(struct replay_validation *validation,
                              uint64_t address, void *data, size_t size)
{
   struct gem_bo *bo =
      intel_hang_replay_find_bo(validation->buffers, address);

   if (!bo || bo->file_offset == -1 || address < bo->offset)
      return false;

   const uint64_t bo_offset = address - bo->offset;

   if (bo_offset > bo->size || size > bo->size - bo_offset)
      return false;

   if (bo->file_offset > validation->file_size ||
       bo_offset > (validation->file_size - bo->file_offset) ||
       size > (validation->file_size - bo->file_offset - bo_offset))
      return false;

   memcpy(data, validation->file_map + bo->file_offset + bo_offset, size);
   return true;
}

enum {
   XE3_RT_DISPATCH_IS_INDIRECT_SHIFT = 0,
   XE3_RT_DISPATCH_LOCAL_X_LOG2_SHIFT = 8,
   XE3_RT_DISPATCH_LOCAL_Y_LOG2_SHIFT = 16,
   XE3_RT_DISPATCH_LOCAL_Z_LOG2_SHIFT = 24,
   XE3_RT_DISPATCH_PARAM_MASK = 0xff,
   XE3_RT_DISPATCH_MAX_IS_INDIRECT = 1,
   XE3_RT_DISPATCH_MAX_LOCAL_LOG2 = 6,
   XE3_RT_GLOBALS_ALIGNMENT = 64,
};

struct xe3_rt_raygen_trampoline_params {
   uint32_t rt_dispatch_globals[2];
   uint32_t raygen_bsr_source[2];
   uint32_t dispatch_params;
   uint32_t reserved[3];
};

/*
 * Mirrors the Xe3 RT_DISPATCH_GLOBALS layout from xe3_rt.xml.
 *
 * The generated GenXML definitions provide packing but not unpacking helpers,
 * while hang replay needs to decode RT_DISPATCH_GLOBALS from captured GPU
 * memory. Keep this layout in sync with the Xe3 GenXML definition.
 */
struct xe3_rt_dispatch_globals {
   uint32_t mem_base_address[2];
   uint32_t call_stack_handler[2];
   uint32_t async_rt_stack_size;
   uint32_t num_dss_rt_stacks;
   uint32_t bvh_hit_miss_fields;
   uint32_t flags;
   uint32_t hit_group_table[2];
   uint32_t miss_group_table[2];
   uint32_t sw_stack_size;
   uint32_t launch_width;
   uint32_t launch_height;
   uint32_t launch_depth;
   uint32_t callable_group_table[2];
   uint32_t callable_group_stride;
   uint32_t resume_shader_table[2];
};

#define XE3_RT_NUM_DSS_RT_STACKS_MASK       0x0000ffffu
#define XE3_RT_BVH_HIT_MISS_FIELDS_MASK     0x1fffffffu
#define XE3_RT_FLAGS_MASK                   0x00000001u
#define XE3_RT_CALLABLE_GROUP_STRIDE_MASK   0x00001fffu

static bool
xe3_rt_dispatch_globals_valid(const struct xe3_rt_dispatch_globals *globals)
{
   return (globals->num_dss_rt_stacks & ~XE3_RT_NUM_DSS_RT_STACKS_MASK) == 0 &&
          (globals->bvh_hit_miss_fields & ~XE3_RT_BVH_HIT_MISS_FIELDS_MASK) == 0 &&
          (globals->flags & ~XE3_RT_FLAGS_MASK) == 0 &&
          (globals->callable_group_stride &
           ~XE3_RT_CALLABLE_GROUP_STRIDE_MASK) == 0;
}

static struct intel_batch_decode_bo
replay_decode_get_bo(void *user_data, bool ppgtt, uint64_t address)
{
   struct replay_validation *validation = user_data;
   struct gem_bo *bo =
      intel_hang_replay_find_bo(validation->buffers, address);

   (void)ppgtt;

   if (!bo || bo->file_offset == -1)
      return (struct intel_batch_decode_bo) {};

   if ((uint64_t)bo->file_offset > validation->file_size ||
       bo->size > validation->file_size - bo->file_offset)
      return (struct intel_batch_decode_bo) {};

   return (struct intel_batch_decode_bo) {
      .addr = bo->offset,
      .size = bo->size,
      .map = validation->file_map + bo->file_offset,
   };
}

static void
replay_validate_compute_walker(void *user_data,
                               const struct intel_compute_walker *walker)
{
   struct replay_validation *validation = user_data;

   if (!walker->emit_inline_parameter ||
       !walker->btd_mode ||
       walker->inline_data_count < ARRAY_SIZE(walker->inline_data))
      return;

   /*
    * The decoder has already interpreted the generation-specific
    * COMPUTE_WALKER layout. The inline data below is Mesa's 32-byte
    * brw_rt_raygen_trampoline_params ABI.
    */
   struct xe3_rt_raygen_trampoline_params trampoline;
   memcpy(&trampoline, walker->inline_data, sizeof(trampoline));

   const uint32_t params = trampoline.dispatch_params;
   const unsigned is_indirect =
      (params >> XE3_RT_DISPATCH_IS_INDIRECT_SHIFT) & XE3_RT_DISPATCH_PARAM_MASK;
   const unsigned local_x_log2 =
      (params >> XE3_RT_DISPATCH_LOCAL_X_LOG2_SHIFT) & XE3_RT_DISPATCH_PARAM_MASK;
   const unsigned local_y_log2 =
      (params >> XE3_RT_DISPATCH_LOCAL_Y_LOG2_SHIFT) & XE3_RT_DISPATCH_PARAM_MASK;
   const unsigned local_z_log2 =
      (params >> XE3_RT_DISPATCH_LOCAL_Z_LOG2_SHIFT) & XE3_RT_DISPATCH_PARAM_MASK;

   if (is_indirect > XE3_RT_DISPATCH_MAX_IS_INDIRECT ||
       local_x_log2 > XE3_RT_DISPATCH_MAX_LOCAL_LOG2 ||
       local_y_log2 > XE3_RT_DISPATCH_MAX_LOCAL_LOG2 ||
       local_z_log2 > XE3_RT_DISPATCH_MAX_LOCAL_LOG2 ||
       trampoline.reserved[0] ||
       trampoline.reserved[1] ||
       trampoline.reserved[2])
      return;

   const uint64_t rt_globals_addr =
      replay_u64_from_dwords(trampoline.rt_dispatch_globals);
   const uint64_t raygen_bsr_source =
      replay_u64_from_dwords(trampoline.raygen_bsr_source);

   if (!rt_globals_addr ||
       (rt_globals_addr & (XE3_RT_GLOBALS_ALIGNMENT - 1)))
      return;

   struct xe3_rt_dispatch_globals rt_globals;

   /*
    * Require RT_DISPATCH_GLOBALS to be captured and its selected hardware
    * bitfields to contain plausible values before interpreting its addresses.
    */
   if (!replay_read_captured_gpu_data(validation,
                                      rt_globals_addr,
                                      &rt_globals,
                                      sizeof(rt_globals))) {
      fprintf(stderr,
              "Failed to read RT_DISPATCH_GLOBALS at 0x%016"PRIx64"\n",
              rt_globals_addr);
      return;
   }

   if (!xe3_rt_dispatch_globals_valid(&rt_globals)) {
      fprintf(stderr,
              "Invalid RT_DISPATCH_GLOBALS at 0x%016"PRIx64"\n",
              rt_globals_addr);
      return;
   }

   uint64_t raygen_bsr = raygen_bsr_source;

   /*
    * For an indirect dispatch, raygen_bsr_source points to memory containing
    * the actual raygen BSR address.  Validate that source before attempting
    * to dereference it from the captured dump.
    */
   if (is_indirect) {
      replay_validate_gpu_reference(validation,
                                    "raygen BSR source",
                                    raygen_bsr_source);

      if (!replay_read_captured_gpu_data(validation,
                                         raygen_bsr_source,
                                         &raygen_bsr,
                                         sizeof(raygen_bsr)))
         raygen_bsr = 0;
   }

   /*
    * Validate the raygen BSR referenced by the trampoline.  If the referenced
    * allocation is absent from the dump, replay no longer has all of the GPU
    * state referenced by the original dispatch.
    */
   replay_validate_gpu_reference(validation,
                                 "raygen BSR",
                                 raygen_bsr);

   /*
    * Validate non-zero shader table references from RT_DISPATCH_GLOBALS.
    * A referenced allocation that is not present in the dump makes the
    * captured state incomplete for replay.
    */
   replay_validate_gpu_reference(
      validation, "hit group table",
      replay_u64_from_dwords(rt_globals.hit_group_table));
   replay_validate_gpu_reference(
      validation, "miss group table",
      replay_u64_from_dwords(rt_globals.miss_group_table));
   replay_validate_gpu_reference(
      validation, "callable group table",
      replay_u64_from_dwords(rt_globals.callable_group_table));
   replay_validate_gpu_reference(
      validation, "resume shader table",
      replay_u64_from_dwords(rt_globals.resume_shader_table));
}

void
intel_hang_replay_validate_rt(int file_fd, size_t file_size,
                              const struct intel_device_info *devinfo,
                              struct util_dynarray *buffers,
                              const struct intel_hang_dump_block_exec *exec)
{
   if (devinfo->kmd_type != INTEL_KMD_TYPE_XE ||
       devinfo->verx10 != 300 ||
       exec->offset == (uint64_t)-1 ||
       file_size == 0)
      return;

   struct gem_bo *exec_bo =
      intel_hang_replay_find_bo(buffers, exec->offset);

   if (!exec_bo ||
       exec_bo->file_offset == -1 ||
       exec->offset < exec_bo->offset) {
      fprintf(stderr,
              "Failed to find captured exec BO at 0x%016"PRIx64"\n",
              exec->offset);
      return;
   }

   const uint64_t exec_delta = exec->offset - exec_bo->offset;

   if (exec_delta >= exec_bo->size) {
      fprintf(stderr,
              "Exec offset 0x%016"PRIx64" is outside captured BO\n",
              exec->offset);
      return;
   }

   void *file_map =
      mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);

   if (file_map == MAP_FAILED) {
      fprintf(stderr, "Failed to mmap hang dump: %s\n", strerror(errno));
      return;
   }

   struct replay_validation validation = {
      .file_map = file_map,
      .file_size = file_size,
      .buffers = buffers,
   };

   FILE *sink = fopen("/dev/null", "w");
   if (!sink) {
      fprintf(stderr, "Failed to open /dev/null: %s\n", strerror(errno));
      munmap(file_map, file_size);
      return;
   }

   struct intel_batch_decode_ctx batch_ctx;
   struct intel_isa_info isa_info;

   intel_decoder_init(&batch_ctx, &isa_info, devinfo, sink,
                      INTEL_BATCH_DECODE_OFFSETS |
                      INTEL_BATCH_DECODE_FULL,
                      NULL,
                      replay_decode_get_bo,
                      NULL,
                      &validation);

   batch_ctx.compute_walker = replay_validate_compute_walker;

   /*
    * Start at the actual submitted EXEC address.  intel_print_batch()
    * follows MI_BATCH_BUFFER_START recursively, so secondary command BOs
    * are visited automatically.
    */
   intel_print_batch(&batch_ctx,
                     (const uint32_t *)(validation.file_map +
                     exec_bo->file_offset + exec_delta),
                     exec_bo->size - exec_delta,
                     exec->offset,
                     false);

   intel_batch_decode_ctx_finish(&batch_ctx);
   fclose(sink);
   munmap(file_map, file_size);

   if (validation.unresolved) {
      if (validation.unresolved_count > 5)
         fprintf(stderr, "  ... additional unresolved GPU references omitted\n");

      fprintf(stderr,
              "\nReplay fidelity cannot be guaranteed.\n"
              "Recapture with INTEL_DEBUG=capture-all "
              "for replay-quality diagnostics.\n"
              "Continuing replay...\n\n");
   }
}
