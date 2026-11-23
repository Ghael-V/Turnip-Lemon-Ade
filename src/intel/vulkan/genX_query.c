/*
 * Copyright © 2015 Intel Corporation
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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "anv_private.h"

#include "util/os_time.h"

#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

#include "ds/intel_tracepoints.h"

#include "anv_internal_kernels.h"
#include "genX_mi_builder.h"

#if GFX_VERx10 >= 125
#define ANV_PIPELINE_STATISTICS_MASK 0x00001fff
#else
#define ANV_PIPELINE_STATISTICS_MASK 0x000007ff
#endif

#include "perf/intel_perf.h"
#include "perf/intel_perf_mdapi.h"
#include "perf/intel_perf_metrics_library.h"
#include "perf/intel_perf_regs.h"

#include "vk_util.h"

static void *
query_slot(struct anv_query_pool *pool, uint32_t query)
{
   return pool->bo->map + query * pool->stride;
}

static struct anv_address
anv_query_address(struct anv_query_pool *pool, uint32_t query)
{
   return (struct anv_address) {
      .bo = pool->bo,
      .offset = query * pool->stride,
   };
}

static void
emit_query_mi_flush_availability(struct anv_cmd_buffer *cmd_buffer,
                                 struct anv_address addr,
                                 bool available)
{
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), flush) {
      flush.PostSyncOperation = WriteImmediateData;
      flush.Address = addr;
      flush.ImmediateData = available;
   }
}

static void
emit_query_mi_availability(struct mi_builder *b,
                           struct anv_address addr,
                           bool available)
{
   mi_store(b, mi_mem64(addr), mi_imm(available));
}

static void
emit_query_pc_availability(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_address addr,
                           bool available)
{
   cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_POST_SYNC_BIT;
   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   genx_batch_emit_pipe_control_write
      (&cmd_buffer->batch, cmd_buffer->device->info,
       cmd_buffer->state.current_pipeline, WriteImmediateData, addr,
       available, 0);
}

/* End of pipe availability */
static void
emit_query_eop_availability(struct anv_cmd_buffer *cmd_buffer,
                            struct anv_address addr,
                            bool available)
{
   switch (cmd_buffer->queue_family->engine_class) {
   case INTEL_ENGINE_CLASS_RENDER:
   case INTEL_ENGINE_CLASS_COMPUTE:
      emit_query_pc_availability(cmd_buffer, addr, available);
      break;

   case INTEL_ENGINE_CLASS_COPY:
   case INTEL_ENGINE_CLASS_VIDEO:
   case INTEL_ENGINE_CLASS_VIDEO_ENHANCE:
      emit_query_mi_flush_availability(cmd_buffer, addr, available);
      break;

   default:
      UNREACHABLE("Invalid engine class");
   }
}

/**
 * VK_INTEL_performance_query layout :
 *
 * ---------------------------------
 * |          query data           |
 * |        (variable size)        |
 * |-------------------------------|
 * |         some padding          |
 * |-------------------------------|
 * |       availability (64b)      |
 * ---------------------------------
 */

static uint32_t
intel_perf_query_availability_size(void)
{
   return 64;
}

static uint32_t
intel_perf_query_availability_offset(struct anv_query_pool *pool)
{
   return pool->stride > intel_perf_query_availability_size() ?
      pool->stride - intel_perf_query_availability_size() :
      0;
}

VkResult genX(CreateQueryPool)(
    VkDevice                                    _device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   const struct anv_physical_device *pdevice = device->physical;
   const VkQueryPoolPerformanceCreateInfoKHR *perf_query_info = NULL;
   struct intel_perf_counter_pass *counter_pass;
   struct intel_perf_query_info **pass_query;
   uint8_t *oag_snapshots = NULL;
   uint32_t n_passes = 0;
   uint32_t data_offset = 0;
   VK_MULTIALLOC(ma);
   VkResult result = VK_SUCCESS;
   void* metrics_library_query_pool = NULL;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);

   /* Query pool slots are made up of some number of 64-bit values packed
    * tightly together. For most query types have the first 64-bit value is
    * the "available" bit which is 0 when the query is unavailable and 1 when
    * it is available. The 64-bit values that follow are determined by the
    * type of query.
    *
    * For performance queries, we have a requirement to align OA reports at
    * 64bytes so we put those first and have the "available" bit behind
    * together with some other counters.
    */
   uint32_t uint64s_per_slot = 0;

   VK_MULTIALLOC_DECL(&ma, struct anv_query_pool, pool, 1);

   VkQueryPipelineStatisticFlags pipeline_statistics = 0;
   switch (pCreateInfo->queryType) {
   case VK_QUERY_TYPE_OCCLUSION:
      /* Occlusion queries have two values: begin and end. */
      uint64s_per_slot = 1 + 2;
      break;
   case VK_QUERY_TYPE_TIMESTAMP:
      /* Timestamps just have the one timestamp value */
      uint64s_per_slot = 1 + 1;
      break;
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      pipeline_statistics = pCreateInfo->pipelineStatistics;
      /* We're going to trust this field implicitly so we need to ensure that
       * no unhandled extension bits leak in.
       */
      pipeline_statistics &= ANV_PIPELINE_STATISTICS_MASK;

      /* Statistics queries have a min and max for every statistic */
      uint64s_per_slot = 1 + 2 * util_bitcount(pipeline_statistics);
      break;
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      /* Transform feedback queries are 4 values, begin/end for
       * written/available.
       */
      uint64s_per_slot = 1 + 4;
      break;
   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      const uint32_t report_size = pdevice->perf->metrics_library.gpu_report_size;
      assert(report_size % sizeof(uint64_t) == 0);
      uint64s_per_slot = DIV_ROUND_UP(report_size, sizeof(uint64_t));
      break;
   }
   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
      const struct intel_perf_query_field_layout *layout =
         &pdevice->perf->query_layout;
      const struct anv_queue_family *queue_family;

      perf_query_info = vk_find_struct_const(pCreateInfo->pNext,
                                             QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR);
      /* Same restriction as in EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR() */
      queue_family = &pdevice->queue.families[perf_query_info->queueFamilyIndex];
      if (!queue_family->supports_perf)
         return vk_error(device, VK_ERROR_UNKNOWN);

      n_passes = intel_perf_get_n_passes(pdevice->perf,
                                         perf_query_info->pCounterIndices,
                                         perf_query_info->counterIndexCount,
                                         NULL);
      vk_multialloc_add(&ma, &counter_pass, struct intel_perf_counter_pass,
                             perf_query_info->counterIndexCount);
      vk_multialloc_add(&ma, &pass_query, struct intel_perf_query_info *,
                             n_passes);
      uint64s_per_slot = 1 /* availability */;
      if (pdevice->perf->oag_global_enable) {
         data_offset = uint64s_per_slot * sizeof(uint64_t);
         uint64s_per_slot += 2 * DIV_ROUND_UP(sizeof(struct anv_oag_boundary), 8);
         vk_multialloc_add(&ma, &oag_snapshots, uint8_t,
                           (pdevice->perf->oa_sample_size * 2 * n_passes * pCreateInfo->queryCount));
      } else {
         /* Align to the requirement of the layout */
         uint64s_per_slot = align(uint64s_per_slot,
                                  DIV_ROUND_UP(layout->alignment, sizeof(uint64_t)));
         data_offset = uint64s_per_slot * sizeof(uint64_t);
         /* Add the query data for begin & end commands */
         uint64s_per_slot += 2 * DIV_ROUND_UP(layout->size, sizeof(uint64_t));
      }
      /* Multiply by the number of passes */
      uint64s_per_slot *= n_passes;
      break;
   }
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
      /* Query has two values: begin and end. */
      uint64s_per_slot = 1 + 2;
      break;
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
      uint64s_per_slot = 1 + 1 /* availability + size (PostbuildInfoCurrentSize, PostbuildInfoCompactedSize) */;
      break;

   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      uint64s_per_slot = 1 + 2 /* availability + size (PostbuildInfoSerializationDesc) */;
      break;

   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
      /* Query has two values: begin and end. */
      uint64s_per_slot = 1 + 2;
      break;

#endif
   case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
      uint64s_per_slot = 1;
      break;
   case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR:
      uint64s_per_slot = 1 + 1; /* availability + length of written bitstream data */
      break;
   default:
      UNREACHABLE("Invalid query type");
   }

   if (!vk_multialloc_zalloc2(&ma, &device->vk.alloc, pAllocator,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT))
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_query_pool_init(&device->vk, &pool->vk, pCreateInfo);
   pool->stride = uint64s_per_slot * sizeof(uint64_t);

   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL) {
      if (pool->stride < intel_perf_query_availability_size()) {
         result = vk_error(device, VK_ERROR_UNKNOWN);
         goto fail;
      }
      metrics_library_query_pool = intel_perf_metrics_library_create_query_pool(pdevice->perf, pool->vk.query_count);

      if (!metrics_library_query_pool) {
         result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }

      pool->metrics_library_query_pool = metrics_library_query_pool;
   }
   else if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR) {
      pool->pass_size = pool->stride / n_passes;
      pool->data_offset = data_offset;
      pool->snapshot_size = (pool->pass_size - data_offset) / 2;
      pool->n_counters = perf_query_info->counterIndexCount;
      pool->counter_pass = counter_pass;
      intel_perf_get_counters_passes(pdevice->perf,
                                     perf_query_info->pCounterIndices,
                                     perf_query_info->counterIndexCount,
                                     pool->counter_pass);
      pool->oag_snapshots = oag_snapshots;
      pool->oag_report_size = pdevice->perf->oa_sample_size;
      pool->n_passes = n_passes;
      pool->pass_query = pass_query;
      intel_perf_get_n_passes(pdevice->perf,
                              perf_query_info->pCounterIndices,
                              perf_query_info->counterIndexCount,
                              pool->pass_query);

      if (pdevice->perf->oag_global_enable) {
         result = anv_oag_alloc_query_ids(device, pool);
         if (result != VK_SUCCESS)
            goto fail;
      }
   } else if (pool->vk.query_type == VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR) {
      const VkVideoProfileInfoKHR* pVideoProfile = vk_find_struct_const(pCreateInfo->pNext, VIDEO_PROFILE_INFO_KHR);
      assert (pVideoProfile);

      pool->codec = pVideoProfile->videoCodecOperation;
   }

   uint64_t size = pool->vk.query_count * (uint64_t)pool->stride;

   /* For KHR_performance_query we need some space in the buffer for a small
    * batch updating ANV_PERF_QUERY_OFFSET_REG.
    *
    * This batch is emitted into a fixed size window with no growth callback,
    * so it must be large enough for everything the preamble writes. In OAG
    * mode it also loads ANV_PERF_QUERY_ID_OFFSET_REG, which does not fit in
    * the 32 bytes the offset register alone needs.
    */
   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR) {
      pool->khr_perf_preamble_stride =
         pdevice->perf->oag_global_enable ? 64 : 32;
      pool->khr_perf_preambles_offset = size;
      size += (uint64_t)pool->n_passes * pool->khr_perf_preamble_stride;
   }

   result = anv_device_alloc_bo(device, "query-pool", size,
                                ANV_BO_ALLOC_MAPPED |
                                ANV_BO_ALLOC_HOST_CACHED_COHERENT |
                                ANV_BO_ALLOC_CAPTURE,
                                0 /* explicit_address */,
                                &pool->bo);
   ANV_DMR_BO_ALLOC(&pool->vk.base, pool->bo, result);
   if (result != VK_SUCCESS)
      goto fail;

   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR) {
      for (uint32_t p = 0; p < pool->n_passes; p++) {
         struct mi_builder b;
         struct anv_batch batch = {
            .start = pool->bo->map + khr_perf_query_preamble_offset(pool, p),
            .end = pool->bo->map + khr_perf_query_preamble_offset(pool, p) + pool->khr_perf_preamble_stride,
         };
         batch.next = batch.start;

         mi_builder_init(&b, device->info, &batch);
         mi_store(&b, mi_reg64(ANV_PERF_QUERY_OFFSET_REG),
                      mi_imm(p * (uint64_t)pool->pass_size));
         if (pool->oag_query_id_base != 0) {
            mi_store(&b, mi_reg32(ANV_PERF_QUERY_ID_OFFSET_REG),
                         mi_imm(p * pool->vk.query_count * 2));
         }
         anv_batch_emit(&batch, GENX(MI_BATCH_BUFFER_END), bbe);

         /* The preamble batch cannot grow, see khr_perf_preamble_stride. */
         assert(batch.next <= batch.end);
      }
   }

   if (pCreateInfo->flags & VK_QUERY_POOL_CREATE_RESET_BIT_KHR) {
      for (uint32_t q = 0; q < pool->vk.query_count; q++) {
         uint64_t *slot = query_slot(pool, q);
         if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL)
            memset((uint8_t *)slot + intel_perf_query_availability_offset(pool), 0, intel_perf_query_availability_size());
         else
            *slot = 0;
      }
   }

   ANV_RMV(query_pool_create, device, pool, false);

   ANV_ADDR_BINDING_REPORT_BO_BIND(device, &pool->vk.base, pool->bo);

   *pQueryPool = anv_query_pool_to_handle(pool);

   return VK_SUCCESS;

 fail:
   if (metrics_library_query_pool) {
      intel_perf_metrics_library_destroy_query_pool(device->physical->perf, metrics_library_query_pool);
   }

   anv_oag_free_query_ids(device, pool);
   vk_free2(&device->vk.alloc, pAllocator, pool);

   return result;
}

void genX(DestroyQueryPool)(
    VkDevice                                    _device,
    VkQueryPool                                 _pool,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_query_pool, pool, _pool);

   if (!pool)
      return;

   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL) {
      if (!intel_perf_metrics_library_destroy_query_pool(device->physical->perf, pool->metrics_library_query_pool))
         vk_error(device, VK_ERROR_UNKNOWN);
   }

   anv_oag_free_query_ids(device, pool);

   ANV_ADDR_BINDING_REPORT_BO_UNBIND(device, &pool->vk.base, pool->bo);
   ANV_RMV(resource_destroy, device, pool);
   ANV_DMR_BO_FREE(&pool->vk.base, pool->bo);
   anv_device_release_bo(device, pool->bo);
   vk_object_free(&device->vk, pAllocator, pool);
}

/**
 * VK_KHR_performance_query layout  :
 *
 * --------------------------------------------
 * |       availability (8b)       | |        |
 * |-------------------------------| |        |
 * |       some padding (see       | |        |
 * | query_field_layout:alignment) | | Pass 0 |
 * |-------------------------------| |        |
 * |           query data          | |        |
 * | (2 * query_field_layout:size) | |        |
 * |-------------------------------|--        | Query 0
 * |       availability (8b)       | |        |
 * |-------------------------------| |        |
 * |       some padding (see       | |        |
 * | query_field_layout:alignment) | | Pass 1 |
 * |-------------------------------| |        |
 * |           query data          | |        |
 * | (2 * query_field_layout:size) | |        |
 * |-------------------------------|-----------
 * |       availability (8b)       | |        |
 * |-------------------------------| |        |
 * |       some padding (see       | |        |
 * | query_field_layout:alignment) | | Pass 0 |
 * |-------------------------------| |        |
 * |           query data          | |        |
 * | (2 * query_field_layout:size) | |        |
 * |-------------------------------|--        | Query 1
 * |               ...             | |        |
 * --------------------------------------------
 */

static struct anv_address
khr_perf_query_availability_address(struct anv_query_pool *pool, uint32_t query, uint32_t pass)
{
   return anv_address_add(
      (struct anv_address) { .bo = pool->bo, },
      khr_perf_query_availability_offset(pool, query, pass));
}

static struct anv_address
khr_perf_query_data_address(struct anv_query_pool *pool, uint32_t query,
                            uint32_t pass, bool end)
{
   return anv_address_add(
      (struct anv_address) { .bo = pool->bo, },
      khr_perf_query_data_offset(pool, query, pass, end));
}

static struct anv_address
khr_perf_query_boundary_address(struct anv_query_pool *pool, uint32_t query,
                                uint32_t pass, bool end)
{
   return anv_address_add(
      (struct anv_address) { .bo = pool->bo, },
      khr_perf_query_boundary_offset(pool, query, pass, end));
}

static bool
khr_perf_query_ensure_relocs(struct anv_cmd_buffer *cmd_buffer)
{
   if (anv_batch_has_error(&cmd_buffer->batch))
      return false;

   if (cmd_buffer->self_mod_locations)
      return true;

   struct anv_device *device = cmd_buffer->device;
   const struct anv_physical_device *pdevice = device->physical;

   cmd_buffer->self_mod_locations =
      vk_alloc(&cmd_buffer->vk.pool->alloc,
               pdevice->n_perf_query_commands * sizeof(*cmd_buffer->self_mod_locations), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (!cmd_buffer->self_mod_locations) {
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   return true;
}

/* Record the address one of the commands of a performance query has to be
 * patched with. @offset is relative to the pool BO, for pass 0; the pass the
 * query is actually replayed with is added at execution time from
 * ANV_PERF_QUERY_OFFSET_REG.
 */
static void
khr_perf_query_push_reloc(struct anv_cmd_buffer *cmd_buffer,
                          struct mi_builder *b, struct anv_query_pool *pool,
                          uint32_t *reloc_idx, uint64_t offset)
{
   struct mi_value addr =
      mi_iadd(b,
              mi_imm(intel_canonical_address(pool->bo->offset + offset)),
              mi_reg64(ANV_PERF_QUERY_OFFSET_REG));

   cmd_buffer->self_mod_locations[(*reloc_idx)++] =
      mi_store_relocated_address_reg64(b, addr);
}

static void
cpu_write_query_result(void *dst_slot, VkQueryResultFlags flags,
                       uint32_t value_index, uint64_t result)
{
   if (flags & VK_QUERY_RESULT_64_BIT) {
      uint64_t *dst64 = dst_slot;
      dst64[value_index] = result;
   } else {
      uint32_t *dst32 = dst_slot;
      dst32[value_index] = result;
   }
}

static bool
query_is_available(struct anv_device *device,
                   struct anv_query_pool *pool,
                   uint32_t query)
{
   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL) {
      /* Dealt with metrics_library */
      return true;
   } else if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR) {
      for (uint32_t p = 0; p < pool->n_passes; p++) {
         volatile uint64_t *slot =
            pool->bo->map + khr_perf_query_availability_offset(pool, query, p);
         if (!slot[0])
            return false;
      }
      return true;
   }

   return *(volatile uint64_t *)query_slot(pool, query);
}

static VkResult
wait_for_available(struct anv_device *device,
                   struct anv_query_pool *pool, uint32_t query)
{
   /* By default we leave a 2s timeout before declaring the device lost. */
   uint64_t rel_timeout = 2 * NSEC_PER_SEC;
   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL) {
      return VK_SUCCESS;
   } else if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR) {
      /* With performance queries, there is an additional 500us reconfiguration
       * time in i915.
       */
      rel_timeout += 500 * 1000;
      /* Additionally a command buffer can be replayed N times to gather data
       * for each of the metric sets to capture all the counters requested.
       */
      rel_timeout *= pool->n_passes;
   }
   uint64_t abs_timeout_ns = os_time_get_absolute_timeout(rel_timeout);

   while (os_time_get_nano() < abs_timeout_ns) {
      if (query_is_available(device, pool, query))
         return VK_SUCCESS;
      VkResult status = vk_device_check_status(&device->vk);
      if (status != VK_SUCCESS)
         return status;
   }

   return vk_device_set_lost(&device->vk, "query timeout");
}

/* OAG global triggered-report support (default on Xe2+).
 *
 * Each begin/end boundary of each (query, pass) writes a distinct marker to
 * OAG_MMIOTRIGGER, which makes the OA unit emit a report carrying that marker,
 * and brackets that trigger with reads of OAG_OATAILPTR. Those two tails
 * delimit the region of the OA buffer the report must have landed in, so
 * resolving a boundary is a bounded search of the (mapped) OA buffer rather
 * than a scan of every report the stream has ever produced.
 *
 * Because the window is captured by the very execution that produced the
 * snapshot, a query slot that is reset and reused can never resolve against a
 * report left behind by an earlier execution.
 */
/* OAG_OASTATUS bits describing a loss of data. They are sticky, so only bits
 * that appear between the two boundaries of a query affect that query, and any
 * of them means its boundary reports cannot be trusted.
 */
#define ANV_OAG_STATUS_REPORT_LOST       (1u << 0)
#define ANV_OAG_STATUS_BUFFER_OVERFLOW   (1u << 1)
#define ANV_OAG_STATUS_MMIO_TRG_Q_FULL   (1u << 6)

#define ANV_OAG_STATUS_FATAL             (ANV_OAG_STATUS_REPORT_LOST | \
                                          ANV_OAG_STATUS_BUFFER_OVERFLOW | \
                                          ANV_OAG_STATUS_MMIO_TRG_Q_FULL)

/* How long to wait for the OA unit to write a triggered report out after the
 * query it delimits has become available. This covers hardware latency only,
 * everything scheduling related has already been waited for.
 */
#define ANV_OAG_RESOLVE_TIMEOUT_NS       (10 * 1000 * 1000)

enum anv_oag_resolve_status {
   /** Both boundary reports were found and accumulated. */
   ANV_OAG_RESOLVED,
   /** Not found yet, the OA unit may still be writing the report. */
   ANV_OAG_PENDING,
   /** The hardware reported dropping data covering this query. */
   ANV_OAG_LOST,
};

static enum anv_oag_resolve_status
anv_oag_accumulate_triggered(struct anv_device *device,
                             struct anv_query_pool *pool,
                             uint32_t query_index, uint32_t pass,
                             const struct intel_perf_query_info *query,
                             struct intel_perf_query_result *result)
{
   /* Snapshots live in host memory and receive the triggered OA reports from
    * anv_oag_resolve_boundary(); the GPU-written boundaries stay in the BO.
    */
   void *begin_snapshot = pool->oag_snapshots +
      khr_perf_query_snapshot_offset(pool, query_index, pass, false);
   void *end_snapshot = pool->oag_snapshots +
      khr_perf_query_snapshot_offset(pool, query_index, pass, true);
   struct anv_oag_boundary *begin_boundary =
      pool->bo->map + khr_perf_query_data_offset(pool, query_index, pass, false);
   struct anv_oag_boundary *end_boundary =
      pool->bo->map + khr_perf_query_data_offset(pool, query_index, pass, true);
   const uint32_t begin_marker =
      anv_oag_query_id(pool, query_index, pass, false);
   const uint32_t end_marker =
      anv_oag_query_id(pool, query_index, pass, true);

   /* Sticky status bits that were not already set when the query started.
    *
    * Loss classification is best effort: OAG_OASTATUS is not writable from
    * userspace batches and the kernel only clears it in stream read() paths
    * this driver never exercises, so a bit that predates the query cannot be
    * attributed to it — and once a bit is set, a genuine loss inside a later
    * query window is equally invisible here. Such a query stays
    * ANV_OAG_PENDING and surfaces through the resolve timeout instead of
    * failing fast.
    */
   const uint32_t lost = end_boundary->oa_status & ~begin_boundary->oa_status;

   simple_mtx_lock(&device->perf_oag.mutex);
   const bool begin_resolved =
      anv_oag_resolve_boundary(device, begin_boundary, begin_snapshot, begin_marker);
   const bool end_resolved =
      anv_oag_resolve_boundary(device, end_boundary, end_snapshot, end_marker);
   simple_mtx_unlock(&device->perf_oag.mutex);

   if (!begin_resolved || !end_resolved) {
      /* OAG_OASTATUS tells us whether waiting any longer can help: a dropped
       * report or an overrun OA buffer means the data is simply gone.
       */
      return (lost & ANV_OAG_STATUS_FATAL) ? ANV_OAG_LOST : ANV_OAG_PENDING;
   }

   if (lost & ANV_OAG_STATUS_FATAL)
      return ANV_OAG_LOST;

   intel_perf_query_result_read_frequencies(result, device->info,
                                            begin_snapshot, end_snapshot);
   intel_perf_query_result_accumulate(result, query,
                                      begin_snapshot, end_snapshot);

   return ANV_OAG_RESOLVED;
}

/* Accumulate every pass of one performance query into @pass_results.
 *
 * A query becoming available only means its triggers retired; the OA unit may
 * still be writing the boundary reports out. That is a hardware latency, not a
 * scheduling one, so it is retried under a much tighter bound than query
 * availability itself.
 */
static enum anv_oag_resolve_status
khr_perf_query_accumulate(struct anv_device *device,
                          struct anv_query_pool *pool, uint32_t query,
                          bool wait,
                          struct intel_perf_query_result *pass_results)
{
   if (pool->oag_query_id_base == 0) {
      /* The OAR snapshots were written directly by the command streamer, so
       * there is never anything to wait for.
       */
      for (uint32_t p = 0; p < pool->n_passes; p++) {
         intel_perf_query_result_clear(&pass_results[p]);
         intel_perf_query_result_accumulate_fields(
            &pass_results[p], pool->pass_query[p],
            pool->bo->map + khr_perf_query_data_offset(pool, query, p, false),
            pool->bo->map + khr_perf_query_data_offset(pool, query, p, true),
            false /* no_oa_accumulate */);
      }
      return ANV_OAG_RESOLVED;
   }

   const uint64_t timeout =
      os_time_get_absolute_timeout(ANV_OAG_RESOLVE_TIMEOUT_NS);
   enum anv_oag_resolve_status resolve;
   STACK_ARRAY(bool, accumulated, pool->n_passes);

   memset(accumulated, 0, pool->n_passes * sizeof(*accumulated));

   while (true) {
      resolve = ANV_OAG_RESOLVED;

      /* Each pass is accumulated exactly once, when it resolves; later
       * iterations only retry the passes still waiting for their reports.
       */
      for (uint32_t p = 0; p < pool->n_passes; p++) {
         if (accumulated[p])
            continue;

         intel_perf_query_result_clear(&pass_results[p]);

         enum anv_oag_resolve_status pass_resolve =
            anv_oag_accumulate_triggered(device, pool, query, p,
                                         pool->pass_query[p],
                                         &pass_results[p]);
         if (pass_resolve == ANV_OAG_RESOLVED)
            accumulated[p] = true;
         else if (resolve != ANV_OAG_LOST)
            resolve = pass_resolve;
      }

      if (resolve != ANV_OAG_PENDING || !wait ||
          os_time_get_nano() >= timeout)
         break;

      /* What is being waited out is OA unit write-out latency, not
       * scheduling; there is nothing to gain from hammering the buffer.
       */
      os_time_sleep(500);
   }

   STACK_ARRAY_FINISH(accumulated);

   return resolve;
}

VkResult genX(GetQueryPoolResults)(
    VkDevice                                    _device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    size_t                                      dataSize,
    void*                                       pData,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);

   assert(
#if GFX_VERx10 >= 125
   pool->vk.query_type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT ||
#endif
   pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION ||
   pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS ||
   pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP ||
   pool->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT ||
   pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL ||
   pool->vk.query_type == VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT ||
   pool->vk.query_type == VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR ||
   pool->vk.query_type == VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR);

   if (vk_device_is_lost(&device->vk))
      return VK_ERROR_DEVICE_LOST;

   if (pData == NULL)
      return VK_SUCCESS;

   /* If stride 0, data is tightly packed */
   if (pool->vk.query_type == VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL && stride == 0)
      stride = device->physical->perf->metrics_library.api_report_size;

   void *data_end = pData + dataSize;

   VkResult status = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      bool available = query_is_available(device, pool, firstQuery + i);

      if (!available && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         status = wait_for_available(device, pool, firstQuery + i);
         if (status != VK_SUCCESS) {
            return status;
         }

         available = true;
      }

      /* From the Vulkan 1.0.42 spec:
       *
       *    "If VK_QUERY_RESULT_WAIT_BIT and VK_QUERY_RESULT_PARTIAL_BIT are
       *    both not set then no result values are written to pData for
       *    queries that are in the unavailable state at the time of the call,
       *    and vkGetQueryPoolResults returns VK_NOT_READY. However,
       *    availability state is still written to pData for those queries if
       *    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT is set."
       *
       * From VK_KHR_performance_query :
       *
       *    "VK_QUERY_RESULT_PERFORMANCE_QUERY_RECORDED_COUNTERS_BIT_KHR specifies
       *     that the result should contain the number of counters that were recorded
       *     into a query pool of type ename:VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR"
       */
      bool write_results = available || (flags & VK_QUERY_RESULT_PARTIAL_BIT);

      uint32_t idx = 0;
      switch (pool->vk.query_type) {
      case VK_QUERY_TYPE_OCCLUSION:
      case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
#if GFX_VERx10 >= 125
      case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
      {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         if (write_results) {
            /* From the Vulkan 1.2.132 spec:
             *
             *    "If VK_QUERY_RESULT_PARTIAL_BIT is set,
             *    VK_QUERY_RESULT_WAIT_BIT is not set, and the query’s status
             *    is unavailable, an intermediate result value between zero and
             *    the final result value is written to pData for that query."
             */
            uint64_t result = available ? slot[2] - slot[1] : 0;
            cpu_write_query_result(pData, flags, idx, result);
         }
         idx++;
         break;
      }

      case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         uint32_t statistics = pool->vk.pipeline_statistics;
         while (statistics) {
            UNUSED uint32_t stat = u_bit_scan(&statistics);
            if (write_results) {
               /* If a query is not available but VK_QUERY_RESULT_PARTIAL_BIT is set, write 0. */
               uint64_t result = available ? slot[idx * 2 + 2] - slot[idx * 2 + 1] : 0;
               cpu_write_query_result(pData, flags, idx, result);
            }
            idx++;
         }
         assert(idx == util_bitcount(pool->vk.pipeline_statistics));
         break;
      }

      case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT: {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         if (write_results) {
            /* If a query is not available but VK_QUERY_RESULT_PARTIAL_BIT is set, write 0. */
            uint64_t result = available ? slot[2] - slot[1] : 0;
            cpu_write_query_result(pData, flags, idx, result);
         }
         idx++;
         if (write_results) {
            /* If a query is not available but VK_QUERY_RESULT_PARTIAL_BIT is set, write 0. */
            uint64_t result = available ? slot[4] - slot[3] : 0;
            cpu_write_query_result(pData, flags, idx, result);
         }
         idx++;
         break;
      }

#if GFX_VERx10 >= 125
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR: {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         if (write_results)
            cpu_write_query_result(pData, flags, idx, slot[1]);
         idx++;
         break;
      }

      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR: {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         if (write_results)
            cpu_write_query_result(pData, flags, idx, slot[2]);
         idx++;
         break;
      }
#endif

      case VK_QUERY_TYPE_TIMESTAMP: {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         if (write_results)
            cpu_write_query_result(pData, flags, idx, slot[1]);
         idx++;
         break;
      }

      case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
         assert((flags & (VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
                          VK_QUERY_RESULT_PARTIAL_BIT)) == 0);
         if (!write_results)
            break;

         STACK_ARRAY(struct intel_perf_query_result, pass_results,
                     pool->n_passes);

         enum anv_oag_resolve_status resolve =
            khr_perf_query_accumulate(device, pool, firstQuery + i,
                                      flags & VK_QUERY_RESULT_WAIT_BIT,
                                      pass_results);

         if (resolve == ANV_OAG_RESOLVED) {
            for (uint32_t p = 0; p < pool->n_passes; p++)
               anv_perf_write_pass_results(device->physical->perf, pool, p,
                                           &pass_results[p], pData);
         } else if (flags & VK_QUERY_RESULT_WAIT_BIT) {
            /* With WAIT_BIT set, VK_NOT_READY is not a legal result, and the
             * boundary reports are not coming: they were either dropped by
             * the OA unit (ANV_OAG_LOST) or never showed up within the
             * resolve timeout. Escalate the same way wait_for_available()
             * does rather than fabricate counter values.
             */
            STACK_ARRAY_FINISH(pass_results);
            return vk_device_set_lost(&device->vk,
                                      "performance query boundary reports "
                                      "unrecoverable");
         } else {
            /* Without WAIT_BIT this is a legal answer. On ANV_OAG_LOST the
             * data will never arrive; the application has to reset the query
             * and execute it again.
             */
            status = VK_NOT_READY;
         }
         STACK_ARRAY_FINISH(pass_results);
         break;
      }

      case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
         if (!write_results)
            break;
         if (!intel_perf_metrics_library_get_query_results(device->physical->perf,
                                                           pool->metrics_library_query_pool,
                                                           pData, firstQuery + i,
                                                           &write_results)) {
            i = queryCount;
            status = VK_ERROR_UNKNOWN;
         }
         break;
      }

      case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
         if (!write_results)
            break;
         const uint32_t *query_data = query_slot(pool, firstQuery + i);
         uint32_t result = available ? *query_data : 0;
         cpu_write_query_result(pData, flags, idx, result);
         break;
      case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR: {
         if (!write_results)
            break;

         /*
          * Slot 0 : Availability.
          * Slot 1 : Bitstream bytes written.
          */
         const uint64_t *slot = query_slot(pool, firstQuery + i);
         /* Set 0 as offset. */
         cpu_write_query_result(pData, flags, idx++, 0);
         cpu_write_query_result(pData, flags, idx++, slot[1]);
         break;
      }

      default:
         UNREACHABLE("invalid pool type");
      }

      if (!write_results)
         status = VK_NOT_READY;

      if (flags & (VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
                   VK_QUERY_RESULT_WITH_STATUS_BIT_KHR))
         cpu_write_query_result(pData, flags, idx, available);

      pData += stride;
      if (pData >= data_end)
         break;
   }

   return status;
}

static void
emit_ps_depth_count(struct anv_cmd_buffer *cmd_buffer,
                    struct anv_address addr)
{
   cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_POST_SYNC_BIT;
   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   bool cs_stall_needed = (GFX_VER == 9 && cmd_buffer->device->info->gt == 4);
   genx_batch_emit_pipe_control_write
      (&cmd_buffer->batch, cmd_buffer->device->info,
       cmd_buffer->state.current_pipeline, WritePSDepthCount, addr, 0,
       ANV_PIPE_DEPTH_STALL_BIT | (cs_stall_needed ? ANV_PIPE_CS_STALL_BIT : 0));
}

/**
 * Goes through a series of consecutive query indices in the given pool
 * setting all element values to 0 and emitting them as available.
 */
static void
emit_zero_queries(struct anv_cmd_buffer *cmd_buffer,
                  struct mi_builder *b, struct anv_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries)
{
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
   case VK_QUERY_TYPE_TIMESTAMP:
      /* These queries are written with a PIPE_CONTROL so clear them using the
       * PIPE_CONTROL as well so we don't have to synchronize between 2 types
       * of operations.
       */
      assert((pool->stride % 8) == 0);
      for (uint32_t i = 0; i < num_queries; i++) {
         struct anv_address slot_addr =
            anv_query_address(pool, first_index + i);

         for (uint32_t qword = 1; qword < (pool->stride / 8); qword++) {
            emit_query_eop_availability(cmd_buffer,
                                        anv_address_add(slot_addr, qword * 8),
                                        false);
         }
         emit_query_eop_availability(cmd_buffer, slot_addr, true);
      }
      break;

   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
      for (uint32_t i = 0; i < num_queries; i++) {
         struct anv_address slot_addr =
            anv_query_address(pool, first_index + i);
         mi_memset(b, anv_address_add(slot_addr, 8), 0, pool->stride - 8);
         emit_query_mi_availability(b, slot_addr, true);
      }
      break;

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
      for (uint32_t i = 0; i < num_queries; i++) {
         for (uint32_t p = 0; p < pool->n_passes; p++) {
            mi_memset(b, khr_perf_query_data_address(pool, first_index + i, p, false),
                         0, 2 * pool->snapshot_size);
            /* No trigger was ever fired for these queries, so their boundary
             * reports will never show up in the OA buffer. Flag them zeroed
             * so the resolve path clears the host snapshots and accumulates
             * them (to zero) instead of waiting for reports that do not
             * exist.
             */
            if (pool->oag_query_id_base != 0) {
               for (uint32_t end = 0; end < 2; end++) {
                  mi_store(b,
                           mi_mem64(anv_address_add(
                              khr_perf_query_boundary_address(
                                 pool, first_index + i, p, end),
                              offsetof(struct anv_oag_boundary, resolved))),
                           mi_imm(ANV_OAG_ZEROED_MAGIC));
               }
            }
            emit_query_mi_availability(b,
                                       khr_perf_query_availability_address(pool, first_index + i, p),
                                       true);
         }
      }
      break;
   }

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      const uint32_t availability_offset = intel_perf_query_availability_offset(pool);
      const uint32_t availability_size = intel_perf_query_availability_size();
      for (uint32_t i = 0; i < num_queries; i++) {
         struct anv_address slot_addr =
            anv_query_address(pool, first_index + i);
         mi_memset(b, slot_addr, 0, availability_offset);
         mi_memset(b, anv_address_add(slot_addr, availability_offset), 1, availability_size);
      }
      break;
   }

   default:
      UNREACHABLE("Unsupported query type");
   }
}

static bool
append_query_copy_flush(struct anv_cmd_buffer *cmd_buffer,
                        struct anv_query_pool *pool,
                        const char *reason)
{
   if (cmd_buffer->state.queries.copy_bits == 0)
      return false;

   anv_add_pending_pipe_bits(cmd_buffer,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             cmd_buffer->state.queries.copy_bits,
                             reason);
   return true;
}

void genX(CmdResetQueryPool)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   const struct anv_physical_device *pdevice = cmd_buffer->device->physical;

   if (append_query_copy_flush(cmd_buffer, pool,
                               "CmdResetQueryPool* flush query copies"))
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   /* Shader clearing is only possible on render/compute when not in protected
    * mode.
    */
   if (anv_cmd_buffer_is_render_or_compute_queue(cmd_buffer) &&
       (cmd_buffer->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT) == 0 &&
       queryCount >= pdevice->drirc.perf.query_clear_with_blorp_threshold) {
      trace_intel_begin_query_clear_blorp(&cmd_buffer->trace);

      anv_cmd_buffer_fill_area(cmd_buffer,
                               anv_query_address(pool, firstQuery),
                               queryCount * pool->stride, 0);

      cmd_buffer->state.queries.clear_bits |= anv_cmd_buffer_shader_query_sync_bits(cmd_buffer);

      trace_intel_end_query_clear_blorp(&cmd_buffer->trace, queryCount);
      return;
   }

   trace_intel_begin_query_clear_cs(&cmd_buffer->trace);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
#endif
      for (uint32_t i = 0; i < queryCount; i++) {
         emit_query_pc_availability(cmd_buffer,
                                    anv_query_address(pool, firstQuery + i),
                                    false);
      }
      break;

   case VK_QUERY_TYPE_TIMESTAMP: {
      for (uint32_t i = 0; i < queryCount; i++) {
         emit_query_eop_availability(cmd_buffer,
                                     anv_query_address(pool, firstQuery + i),
                                     false);
      }

      /* Add a CS stall here to make sure the PIPE_CONTROL above has
       * completed. Otherwise some timestamps written later with MI_STORE_*
       * commands might race with the PIPE_CONTROL in the loop above.
       */
      anv_add_pending_pipe_bits(cmd_buffer,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                ANV_PIPE_CS_STALL_BIT,
                                "vkCmdResetQueryPool of timestamps");
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);
      break;
   }

   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR:
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
   {
      struct mi_builder b;
      mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);

      for (uint32_t i = 0; i < queryCount; i++)
         emit_query_mi_availability(&b, anv_query_address(pool, firstQuery + i), false);
      break;
   }

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
      struct mi_builder b;
      mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);

      for (uint32_t i = 0; i < queryCount; i++) {
         for (uint32_t p = 0; p < pool->n_passes; p++) {
            if (pool->oag_query_id_base != 0) {
               /* Drop the whole boundary trailer, not just the "resolved"
                * flag: a stale OATAIL window must never be able to resolve a
                * snapshot that the next execution has not written yet.
                */
               for (uint32_t end = 0; end < 2; end++) {
                  mi_memset(&b,
                            khr_perf_query_boundary_address(pool,
                                                            firstQuery + i, p,
                                                            end),
                            0, sizeof(struct anv_oag_boundary));
               }
            }
            emit_query_mi_availability(
               &b,
               khr_perf_query_availability_address(pool, firstQuery + i, p),
               false);
         }
      }
      break;
   }

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      struct mi_builder b;
      mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);

      const uint32_t availability_offset = intel_perf_query_availability_offset(pool);
      const uint32_t availability_size = intel_perf_query_availability_size();

      for (uint32_t i = 0; i < queryCount; i++) {
         struct anv_address slot_addr =
            anv_query_address(pool, firstQuery + i);
         mi_memset(&b, anv_address_add(slot_addr, availability_offset), 0, availability_size);
      }
      break;
   }
   case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
      for (uint32_t i = 0; i < queryCount; i++)
         emit_query_mi_flush_availability(cmd_buffer, anv_query_address(pool, firstQuery + i), false);
      break;
   default:
      UNREACHABLE("Unsupported query type");
   }

   trace_intel_end_query_clear_cs(&cmd_buffer->trace, queryCount);

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_QUERY;
}

void genX(ResetQueryPool)(
    VkDevice                                    _device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
   case VK_QUERY_TYPE_TIMESTAMP:
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
   case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
   case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR:
      for (uint32_t i = 0; i < queryCount; i++) {
         uint64_t *slot = query_slot(pool, firstQuery + i);
         *slot = 0;
      }
      break;
   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR:
      for (uint32_t i = 0; i < queryCount; i++) {
         uint32_t q = firstQuery + i;
         for (uint32_t p = 0; p < pool->n_passes; p++) {
            uint64_t *slot = pool->bo->map +
               khr_perf_query_availability_offset(pool, q, p);
            *slot = 0;
            if (pool->oag_query_id_base != 0) {
               for (uint32_t end = 0; end < 2; end++) {
                  memset(pool->bo->map +
                         khr_perf_query_boundary_offset(pool, q, p, end),
                         0, sizeof(struct anv_oag_boundary));
               }
            }
         }
      }
      break;
   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      const uint32_t availability_offset = intel_perf_query_availability_offset(pool);
      const uint32_t availability_size = intel_perf_query_availability_size();
      for (uint32_t i = 0; i < queryCount; i++) {
         uint8_t *slot = query_slot(pool, firstQuery + i);
         memset(slot + availability_offset, 0, availability_size);
      }
      break;
   }
   default:
      UNREACHABLE("Unsupported query type");
   }
}

static const uint32_t vk_pipeline_stat_to_reg[] = {
   GENX(IA_VERTICES_COUNT_num),
   GENX(IA_PRIMITIVES_COUNT_num),
   GENX(VS_INVOCATION_COUNT_num),
   GENX(GS_INVOCATION_COUNT_num),
   GENX(GS_PRIMITIVES_COUNT_num),
   GENX(CL_INVOCATION_COUNT_num),
   GENX(CL_PRIMITIVES_COUNT_num),
   GENX(PS_INVOCATION_COUNT_num),
   GENX(HS_INVOCATION_COUNT_num),
   GENX(DS_INVOCATION_COUNT_num),
   GENX(CS_INVOCATION_COUNT_num),
#if GFX_VERx10 >= 125
   GENX(TASK_INVOCATION_COUNT_num),
   GENX(MESH_INVOCATION_COUNT_num)
#endif
};

static void
emit_pipeline_stat(struct mi_builder *b, uint32_t stat,
                   struct anv_address addr)
{
   STATIC_ASSERT(ANV_PIPELINE_STATISTICS_MASK ==
                 (1 << ARRAY_SIZE(vk_pipeline_stat_to_reg)) - 1);

   assert(stat < ARRAY_SIZE(vk_pipeline_stat_to_reg));
   mi_store(b, mi_mem64(addr), mi_reg64(vk_pipeline_stat_to_reg[stat]));
}

static void
emit_xfb_query(struct mi_builder *b, uint32_t stream,
               struct anv_address addr)
{
   assert(stream < MAX_XFB_STREAMS);

   mi_store(b, mi_mem64(anv_address_add(addr, 0)),
               mi_reg64(GENX(SO_NUM_PRIMS_WRITTEN0_num) + stream * 8));
   mi_store(b, mi_mem64(anv_address_add(addr, 16)),
               mi_reg64(GENX(SO_PRIM_STORAGE_NEEDED0_num) + stream * 8));
}

/* OAG registers read back from the command streamer around a triggered report.
 *
 * On xe these are whitelisted for userspace RING_FORCE_TO_NONPRIV batches on
 * RENDER/COMPUTE while a sampling OA stream is open: OAG_MMIOTRIGGER is
 * read/write, and OAG_OAHEADPTR is readable with a 16 byte range which also
 * covers OAG_OATAILPTR and OAG_OABUFFER. OAG_OASTATUS is readable too.
 */
#define OAG_TAIL_REG        0xDB04
#define OAG_BUFFER_REG      0xDB08
#define OAG_STATUS_REG      0xDAFC

/* OAG on-demand report trigger register (OAG_MMIOTRIGGER). Writing a marker to
 * it makes the OA unit emit a report at that exact point in the command
 * stream, carrying the marker in the report context-id field. Used to bracket
 * a query with exact boundary reports for the OAG performance-query resolve.
 */
#define OAG_MMIOTRIGGER_REG 0xDB1C

/* The register stores that make up one OAG boundary, in emission order.
 *
 * Each one needs a self-modifying relocation, so both the batch emission
 * (emit_perf_khr_oag_boundary) and the relocation setup
 * (emit_perf_khr_oag_relocs) walk this table to stay in lockstep.
 */
static const struct {
   uint32_t reg;
   size_t   offset;
} oag_boundary_stores[] = {
   { OAG_TAIL_REG,   offsetof(struct anv_oag_boundary, tail_pre)  },
   { OAG_STATUS_REG, offsetof(struct anv_oag_boundary, oa_status) },
   /* The trigger is fired here, between the two OATAIL samples. */
   { OAG_TAIL_REG,   offsetof(struct anv_oag_boundary, tail_post) },
   { OAG_BUFFER_REG, offsetof(struct anv_oag_boundary, oa_buffer) },
};

/* Index in oag_boundary_stores[] the MMIO trigger is emitted in front of. */
#define OAG_TRIGGER_STORE_IDX 2

/* Emit the OAG boundary of a performance query: sample the OA tail and status,
 * fire the trigger, then sample the tail again and record the OA buffer base.
 * The two tails delimit the region of the OA buffer the triggered report must
 * have landed in.
 */
static void
emit_perf_khr_oag_boundary(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_query_pool *pool,
                           struct mi_builder *b,
                           struct anv_address query_addr,
                           uint32_t query, bool end)
{
   STATIC_ASSERT(ARRAY_SIZE(oag_boundary_stores) == ANV_OAG_BOUNDARY_STORES);

   for (uint32_t i = 0; i < ARRAY_SIZE(oag_boundary_stores); i++) {
      if (i == OAG_TRIGGER_STORE_IDX) {
         mi_store(b, mi_reg32(OAG_MMIOTRIGGER_REG),
                     mi_iadd(b, mi_imm(anv_oag_query_id(pool, query, 0, end)),
                                mi_reg32(ANV_PERF_QUERY_ID_OFFSET_REG)));
      }

      void *dws = anv_batch_emitn(&cmd_buffer->batch,
                                  GENX(MI_STORE_REGISTER_MEM_length),
                                  GENX(MI_STORE_REGISTER_MEM),
                                  .RegisterAddress = oag_boundary_stores[i].reg,
                                  .MemoryAddress = query_addr /* overwritten */);
      mi_resolve_relocated_address_token(
         b, cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
         dws + GENX(MI_STORE_REGISTER_MEM_MemoryAddress_start) / 8);
   }
}

static bool
append_query_clear_flush(struct anv_cmd_buffer *cmd_buffer,
                         struct anv_query_pool *pool,
                         const char *reason)
{
   if (cmd_buffer->state.queries.clear_bits == 0)
      return false;

   anv_add_pending_pipe_bits(cmd_buffer,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             cmd_buffer->state.queries.clear_bits,
                             reason);
   return true;
}

void genX(CmdBeginQueryIndexedEXT)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags,
    uint32_t                                    index)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   if (append_query_clear_flush(cmd_buffer, pool,
                                "CmdBeginQuery* flush query clears"))
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);
   const uint32_t mocs = anv_mocs_for_address(cmd_buffer->device, &query_addr);
   mi_builder_set_mocs(&b, mocs);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      cmd_buffer->state.gfx.n_occlusion_queries++;
      cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_OCCLUSION_QUERY_ACTIVE;
      emit_ps_depth_count(cmd_buffer, anv_address_add(query_addr, 8));
      break;

   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      mi_store(&b, mi_mem64(anv_address_add(query_addr, 8)),
                   mi_reg64(GENX(CL_INVOCATION_COUNT_num)));
      break;

#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      mi_store(&b, mi_mem64(anv_address_add(query_addr, 8)),
                   mi_reg64(GENX(MESH_PRIMITIVE_COUNT_num)));
      break;
#endif

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      /* TODO: This might only be necessary for certain stats */
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);

      uint32_t statistics = pool->vk.pipeline_statistics;
      uint32_t offset = 8;
      while (statistics) {
         uint32_t stat = u_bit_scan(&statistics);
         emit_pipeline_stat(&b, stat, anv_address_add(query_addr, offset));
         offset += 16;
      }
      break;
   }

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      emit_xfb_query(&b, index, anv_address_add(query_addr, 8));
      break;

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
      if (!khr_perf_query_ensure_relocs(cmd_buffer))
         return;

      const struct anv_physical_device *pdevice = cmd_buffer->device->physical;
      const struct intel_perf_query_field_layout *layout = &pdevice->perf->query_layout;
      const bool oag = pool->oag_query_id_base != 0;

      uint32_t reloc_idx = 0;
      for (uint32_t end = 0; end < 2; end++) {
         /* In OAG mode the MI_RPC field is replaced by a trigger write, which
          * needs no relocation, and the stores bracketing it. The SRM fields
          * are not emitted at all: they read the context-relative OAR
          * sub-unit, which is zero for GT-wide counters.
          */
         if (oag) {
            const uint64_t boundary_offset =
               khr_perf_query_data_offset(pool, query, 0 /* pass */, end);

            for (uint32_t s = 0; s < ARRAY_SIZE(oag_boundary_stores); s++) {
               khr_perf_query_push_reloc(cmd_buffer, &b, pool, &reloc_idx,
                                         boundary_offset +
                                         oag_boundary_stores[s].offset);
            }
         } else {
            const uint64_t data_offset =
               khr_perf_query_data_offset(pool, query, 0 /* pass */, end);

            for (uint32_t r = 0; r < layout->n_fields; r++) {
               const struct intel_perf_query_field *field =
                  &layout->fields[end ? r : (layout->n_fields - 1 - r)];

               khr_perf_query_push_reloc(cmd_buffer, &b, pool, &reloc_idx,
                                         data_offset + field->location);

               /* 64bit registers are stored with two MI_STORE_REGISTER_MEM. */
               if (field->type != INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC &&
                   field->size == 8) {
                  khr_perf_query_push_reloc(cmd_buffer, &b, pool, &reloc_idx,
                                            data_offset + field->location + 4);
               }
            }
         }
      }

      khr_perf_query_push_reloc(cmd_buffer, &b, pool, &reloc_idx,
                                khr_perf_query_availability_offset(pool, query,
                                                                   0 /* pass */));

      assert(reloc_idx == pdevice->n_perf_query_commands);

      const struct intel_device_info *devinfo = cmd_buffer->device->info;
      const enum intel_engine_class engine_class = cmd_buffer->queue_family->engine_class;
      mi_self_mod_barrier(&b, devinfo->engine_class_prefetch[engine_class]);

      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      cmd_buffer->perf_query_pool = pool;

      cmd_buffer->perf_reloc_idx = 0;
      for (uint32_t r = 0; r < layout->n_fields; r++) {
         const struct intel_perf_query_field *field =
            &layout->fields[layout->n_fields - 1 - r];
         void *dws;

         /* In OAG mode only the MI_RPC field is emitted, replaced by an OAG
          * MMIO trigger and its delimiting register stores; the dead
          * OAR-relative SRM stores are skipped.
          */
         if (oag && field->type != INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC)
            continue;

         switch (field->type) {
         case INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC:
            if (oag) {
               emit_perf_khr_oag_boundary(cmd_buffer, pool, &b,
                                          anv_address_add(query_addr, pool->data_offset),
                                          query, false /* end */);
            } else {
               dws = anv_batch_emitn(&cmd_buffer->batch,
                                     GENX(MI_REPORT_PERF_COUNT_length),
                                     GENX(MI_REPORT_PERF_COUNT),
                                     .MemoryAddress = query_addr /* Will be overwritten */);
               mi_resolve_relocated_address_token(
                  &b,
                  cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
                  dws + GENX(MI_REPORT_PERF_COUNT_MemoryAddress_start) / 8);
            }
            break;

         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_PERFCNT:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_RPSTAT:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_A:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_B:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_C:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_PEC:
            dws =
               anv_batch_emitn(&cmd_buffer->batch,
                               GENX(MI_STORE_REGISTER_MEM_length),
                               GENX(MI_STORE_REGISTER_MEM),
                               .RegisterAddress = field->mmio_offset,
                               .MemoryAddress = query_addr /* Will be overwritten */ );
            mi_resolve_relocated_address_token(
               &b,
               cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
               dws + GENX(MI_STORE_REGISTER_MEM_MemoryAddress_start) / 8);
            if (field->size == 8) {
               dws =
                  anv_batch_emitn(&cmd_buffer->batch,
                                  GENX(MI_STORE_REGISTER_MEM_length),
                                  GENX(MI_STORE_REGISTER_MEM),
                                  .RegisterAddress = field->mmio_offset + 4,
                                  .MemoryAddress = query_addr /* Will be overwritten */ );
               mi_resolve_relocated_address_token(
                  &b,
                  cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
                  dws + GENX(MI_STORE_REGISTER_MEM_MemoryAddress_start) / 8);
            }
            break;

         default:
            UNREACHABLE("Invalid query field");
            break;
         }
      }
      break;
   }

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      bool success = false;
      uint32_t cmds_size = 0;
      const uint64_t query_pool_gpu_addr = anv_address_physical(anv_query_address(pool, 0));
      void* query_pool_cpu_addr = query_slot(pool, 0);
      if (intel_perf_metrics_library_get_perf_query_cmds(cmd_buffer->device->physical->perf,
                                                         pool->metrics_library_query_pool,
                                                         query_pool_gpu_addr,
                                                         query_pool_cpu_addr,
                                                         query,
                                                         cmd_buffer->intel_perf_marker,
                                                         true,
                                                         NULL,
                                                         &cmds_size)) {
         assert(cmds_size % 4 == 0);
         void* cmds = anv_batch_emit_dwords(&cmd_buffer->batch, cmds_size / 4);

         success = cmds && intel_perf_metrics_library_get_perf_query_cmds(cmd_buffer->device->physical->perf,
                                                                          pool->metrics_library_query_pool,
                                                                          query_pool_gpu_addr,
                                                                          query_pool_cpu_addr,
                                                                          query,
                                                                          cmd_buffer->intel_perf_marker,
                                                                          true,
                                                                          cmds,
                                                                          &cmds_size);

         if (!success)
            anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      break;
   }
   case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
      emit_query_mi_flush_availability(cmd_buffer, query_addr, false);
      break;
   case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR:
      emit_query_mi_availability(&b, query_addr, false);
      break;
   default:
      UNREACHABLE("");
   }

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_QUERY;
}

void genX(CmdEndQueryIndexedEXT)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    uint32_t                                    index)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);
   /* Required before emitting MI_MATH on Gfx12.5+, where its MOCS field must
    * not be zero. The OAG boundary of a performance query computes its trigger
    * id with mi_iadd().
    */
   const uint32_t mocs = anv_mocs_for_address(cmd_buffer->device, &query_addr);
   mi_builder_set_mocs(&b, mocs);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      emit_ps_depth_count(cmd_buffer, anv_address_add(query_addr, 16));
      emit_query_pc_availability(cmd_buffer, query_addr, true);
      cmd_buffer->state.gfx.n_occlusion_queries--;
      cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_OCCLUSION_QUERY_ACTIVE;
      break;

   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
      /* Ensure previous commands have completed before capturing the register
       * value.
       */
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);

      mi_store(&b, mi_mem64(anv_address_add(query_addr, 16)),
                   mi_reg64(GENX(CL_INVOCATION_COUNT_num)));
      emit_query_mi_availability(&b, query_addr, true);
      break;

#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      mi_store(&b, mi_mem64(anv_address_add(query_addr, 16)),
                   mi_reg64(GENX(MESH_PRIMITIVE_COUNT_num)));
      emit_query_mi_availability(&b, query_addr, true);
      break;
#endif

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      /* TODO: This might only be necessary for certain stats */
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);

      uint32_t statistics = pool->vk.pipeline_statistics;
      uint32_t offset = 16;
      while (statistics) {
         uint32_t stat = u_bit_scan(&statistics);
         emit_pipeline_stat(&b, stat, anv_address_add(query_addr, offset));
         offset += 16;
      }

      emit_query_mi_availability(&b, query_addr, true);
      break;
   }

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      emit_xfb_query(&b, index, anv_address_add(query_addr, 16));
#if GFX_VER == 11
      /* Running the following CTS pattern on ICL will likely report a failure :
       *
       * dEQP-VK.transform_feedback.primitives_generated_query.get.queue_reset.32bit.geom.*
       *
       * If you dump the returned values in genX(GetQueryPoolResults)(), you
       * will notice that the last 64bit value is 0 and rereading the value
       * once more will return a non-zero value. This seems to indicate that
       * the memory writes are not ordered somehow... Otherwise the
       * availability write below would ensure the previous writes above have
       * completed.
       *
       * So as a workaround, we stall CS to make sure the previous writes have
       * landed before emitting the availability.
       */
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT);
#endif
      emit_query_mi_availability(&b, query_addr, true);
      break;

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR: {
      genx_batch_emit_pipe_control(&cmd_buffer->batch,
                                   cmd_buffer->device->info,
                                   cmd_buffer->state.current_pipeline,
                                   ANV_PIPE_CS_STALL_BIT |
                                   ANV_PIPE_STALL_AT_SCOREBOARD_BIT);
      cmd_buffer->perf_query_pool = pool;

      if (!khr_perf_query_ensure_relocs(cmd_buffer))
         return;

      const struct anv_physical_device *pdevice = cmd_buffer->device->physical;
      const struct intel_perf_query_field_layout *layout = &pdevice->perf->query_layout;
      const bool oag = pool->oag_query_id_base != 0;

      void *dws;
      for (uint32_t r = 0; r < layout->n_fields; r++) {
         const struct intel_perf_query_field *field = &layout->fields[r];

         /* In OAG mode only the MI_RPC field is emitted, replaced by an OAG
          * MMIO trigger and its delimiting register stores; the dead
          * OAR-relative SRM stores are skipped.
          */
         if (oag && field->type != INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC)
            continue;

         switch (field->type) {
         case INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC:
            if (oag) {
               emit_perf_khr_oag_boundary(cmd_buffer, pool, &b,
                                          anv_address_add(query_addr, pool->data_offset),
                                          query, true /* end */);
            } else {
               dws = anv_batch_emitn(&cmd_buffer->batch,
                                     GENX(MI_REPORT_PERF_COUNT_length),
                                     GENX(MI_REPORT_PERF_COUNT),
                                     .MemoryAddress = query_addr /* Will be overwritten */);
               mi_resolve_relocated_address_token(
                  &b,
                  cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
                  dws + GENX(MI_REPORT_PERF_COUNT_MemoryAddress_start) / 8);
            }
            break;

         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_PERFCNT:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_RPSTAT:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_A:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_B:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_C:
         case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_PEC:
            dws =
               anv_batch_emitn(&cmd_buffer->batch,
                               GENX(MI_STORE_REGISTER_MEM_length),
                               GENX(MI_STORE_REGISTER_MEM),
                               .RegisterAddress = field->mmio_offset,
                               .MemoryAddress = query_addr /* Will be overwritten */ );
            mi_resolve_relocated_address_token(
               &b,
               cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
               dws + GENX(MI_STORE_REGISTER_MEM_MemoryAddress_start) / 8);
            if (field->size == 8) {
               dws =
                  anv_batch_emitn(&cmd_buffer->batch,
                                  GENX(MI_STORE_REGISTER_MEM_length),
                                  GENX(MI_STORE_REGISTER_MEM),
                                  .RegisterAddress = field->mmio_offset + 4,
                                  .MemoryAddress = query_addr /* Will be overwritten */ );
               mi_resolve_relocated_address_token(
                  &b,
                  cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
                  dws + GENX(MI_STORE_REGISTER_MEM_MemoryAddress_start) / 8);
            }
            break;

         default:
            UNREACHABLE("Invalid query field");
            break;
         }
      }

      dws =
         anv_batch_emitn(&cmd_buffer->batch,
                         GENX(MI_STORE_DATA_IMM_length),
                         GENX(MI_STORE_DATA_IMM),
                         .ImmediateData = true);
      mi_resolve_relocated_address_token(
         &b,
         cmd_buffer->self_mod_locations[cmd_buffer->perf_reloc_idx++],
         dws + GENX(MI_STORE_DATA_IMM_Address_start) / 8);

      assert(cmd_buffer->perf_reloc_idx == pdevice->n_perf_query_commands);
      break;
   }

   case VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL: {
      bool success = false;
      uint32_t cmds_size = 0;
      const uint64_t query_pool_gpu_addr = anv_address_physical(anv_query_address(pool, 0));
      void* query_pool_cpu_addr = query_slot(pool, 0);
      if (intel_perf_metrics_library_get_perf_query_cmds(cmd_buffer->device->physical->perf,
                                                         pool->metrics_library_query_pool,
                                                         query_pool_gpu_addr,
                                                         query_pool_cpu_addr,
                                                         query,
                                                         cmd_buffer->intel_perf_marker,
                                                         false,
                                                         NULL,
                                                         &cmds_size)) {
         assert(cmds_size % 4 == 0);
         void* cmds = anv_batch_emit_dwords(&cmd_buffer->batch, cmds_size / 4);

         success = cmds && intel_perf_metrics_library_get_perf_query_cmds(cmd_buffer->device->physical->perf,
                                                                          pool->metrics_library_query_pool,
                                                                          query_pool_gpu_addr,
                                                                          query_pool_cpu_addr,
                                                                          query,
                                                                          cmd_buffer->intel_perf_marker,
                                                                          false,
                                                                          cmds,
                                                                          &cmds_size);

         if (!success)
            anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      break;
   }
   case VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR:
      emit_query_mi_flush_availability(cmd_buffer, query_addr, true);
      break;

#if GFX_VER < 11
#define MFC_BITSTREAM_BYTECOUNT_FRAME_REG       0x128A0
#define HCP_BITSTREAM_BYTECOUNT_FRAME_REG       0x1E9A0
#elif GFX_VER >= 11
#define MFC_BITSTREAM_BYTECOUNT_FRAME_REG       0x1C08A0
#define HCP_BITSTREAM_BYTECOUNT_FRAME_REG       0x1C28A0
#endif

   case VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR: {
      struct mi_value val;
      if (pool->codec & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR) {
         val = mi_reg32(MFC_BITSTREAM_BYTECOUNT_FRAME_REG);
      } else if (pool->codec & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR) {
         val = mi_reg32(HCP_BITSTREAM_BYTECOUNT_FRAME_REG);
      } else if (pool->codec & VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR) {
         /* AV1 BITSTREAM_BYTECOUNT_TILE is per-tile; the frame total (all
          * tiles) is the running sum accumulated into the encode scratch
          * dword during CmdEncodeVideoKHR. */
         struct anv_video_session *vid = cmd_buffer->video.vid;
         struct anv_address scratch = {
            vid->vid_mem[ANV_VID_MEM_AV1_ENCODE_TILE_BITSTREAM_ACCUM].mem->bo,
            vid->vid_mem[ANV_VID_MEM_AV1_ENCODE_TILE_BITSTREAM_ACCUM].offset
         };
         val = mi_mem32(scratch);
      } else {
         UNREACHABLE("Invalid codec operation");
      }
      mi_store(&b, mi_mem64(anv_address_add(query_addr, 8)), val);
      emit_query_mi_availability(&b, query_addr, true);

      break;
   }
   default:
      UNREACHABLE("");
   }

   /* When multiview is active the spec requires that N consecutive query
    * indices are used, where N is the number of active views in the subpass.
    * The spec allows that we only write the results to one of the queries
    * but we still need to manage result availability for all the query indices.
    * Since we only emit a single query for all active views in the
    * first index, mark the other query indices as being already available
    * with result 0.
    */
   if (cmd_buffer->state.gfx.view_mask) {
      const uint32_t num_queries =
         util_bitcount(cmd_buffer->state.gfx.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd_buffer, &b, pool, query + 1, num_queries - 1);
   }

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_QUERY;
}

#define TIMESTAMP 0x2358

void genX(CmdWriteTimestamp2)(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlags2                       stage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   assert(pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP);

   /* Anything bottom-of-pipe, request a post-sync */
   if (stage != VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
      cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_POST_SYNC_BIT;

   append_query_clear_flush(cmd_buffer, pool,
                            "CmdWriteTimestamp flush query clears");

   /* Always flush, even for top-of-pipe there might be a barrier that needs
    * executing before we take the timestamp.
    */
   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);

   if (stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) {
      mi_store(&b, mi_mem64(anv_address_add(query_addr, 8)),
                   mi_reg64(TIMESTAMP));
      emit_query_mi_availability(&b, query_addr, true);
   } else {
      bool cs_stall_needed =
         (GFX_VER == 9 && cmd_buffer->device->info->gt == 4);

      if (anv_cmd_buffer_is_blitter_queue(cmd_buffer) ||
          anv_cmd_buffer_is_video_queue(cmd_buffer)) {
         /* Wa_16018063123 - emit fast color dummy blit before MI_FLUSH_DW. */
         if (INTEL_WA_16018063123_GFX_VER &&
             anv_cmd_buffer_is_blitter_queue(cmd_buffer)) {
            genX(batch_emit_fast_color_dummy_blit)(&cmd_buffer->batch,
                                                   cmd_buffer->device);
         }
         anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), dw) {
            dw.Address = anv_address_add(query_addr, 8);
            dw.PostSyncOperation = WriteTimestamp;
         }
         emit_query_mi_flush_availability(cmd_buffer, query_addr, true);
      } else {
         genx_batch_emit_pipe_control_write
            (&cmd_buffer->batch, cmd_buffer->device->info,
             cmd_buffer->state.current_pipeline, WriteTimestamp,
             anv_address_add(query_addr, 8), 0,
             cs_stall_needed ? ANV_PIPE_CS_STALL_BIT : 0);
         emit_query_pc_availability(cmd_buffer, query_addr, true);
      }

   }


   /* When multiview is active the spec requires that N consecutive query
    * indices are used, where N is the number of active views in the subpass.
    * The spec allows that we only write the results to one of the queries
    * but we still need to manage result availability for all the query indices.
    * Since we only emit a single query for all active views in the
    * first index, mark the other query indices as being already available
    * with result 0.
    */
   if (cmd_buffer->state.gfx.view_mask) {
      const uint32_t num_queries =
         util_bitcount(cmd_buffer->state.gfx.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd_buffer, &b, pool, query + 1, num_queries - 1);
   }

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_QUERY;
}

#define MI_PREDICATE_SRC0    0x2400
#define MI_PREDICATE_SRC1    0x2408
#define MI_PREDICATE_RESULT  0x2418

static void
gpu_write_query_on_predicate(struct anv_cmd_buffer *cmd_buffer,
                             struct mi_builder *b,
                             struct anv_address base_store_addr,
                             uint32_t store_index,
                             bool store_64bits,
                             struct mi_value store_val,
                             uint64_t predicate_value)
{
   mi_store(b, mi_reg64(MI_PREDICATE_SRC1), mi_imm(predicate_value));
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_PREDICATE), mip) {
      mip.LoadOperation    = LOAD_LOAD;
         mip.CombineOperation = COMBINE_SET;
         mip.CompareOperation = COMPARE_SRCS_EQUAL;
   }

   if (store_64bits) {
      struct anv_address addr = anv_address_add(base_store_addr, store_index * 8);
      mi_store_if(b, mi_mem64(addr), store_val);
   } else {
      struct anv_address addr = anv_address_add(base_store_addr, store_index * 4);
      mi_store_if(b, mi_mem32(addr), store_val);
   }
}

static void
gpu_write_query_result(struct anv_cmd_buffer *cmd_buffer,
                       struct mi_builder *b,
                       struct anv_address poll_addr,
                       struct anv_address dst_addr,
                       VkQueryResultFlags flags,
                       uint32_t value_index,
                       struct mi_value query_result)
{
   /* Like in the case of vkGetQueryPoolResults, if the query is unavailable
    * and the VK_QUERY_RESULT_PARTIAL_BIT flag is set, conservatively write 0
    * as the query result. If the VK_QUERY_RESULT_PARTIAL_BIT isn't set, don't
    * write any value.
    */
   if (flags & VK_QUERY_RESULT_PARTIAL_BIT) {
      gpu_write_query_on_predicate(cmd_buffer, b, dst_addr, value_index,
                                   flags & VK_QUERY_RESULT_64_BIT,
                                   query_result, 1);
      gpu_write_query_on_predicate(cmd_buffer, b, dst_addr, value_index,
                                   flags & VK_QUERY_RESULT_64_BIT,
                                   mi_imm(0), 0);
   } else {
      if (flags & VK_QUERY_RESULT_64_BIT) {
         struct anv_address res_addr = anv_address_add(dst_addr, value_index * 8);
         mi_store(b, mi_mem64(res_addr), query_result);
      } else {
         struct anv_address res_addr = anv_address_add(dst_addr, value_index * 4);
         mi_store(b, mi_mem32(res_addr), query_result);
      }
   }
}

static struct mi_value
compute_query_result(struct mi_builder *b, struct anv_address addr)
{
   return mi_isub(b, mi_mem64(anv_address_add(addr, 8)),
                     mi_mem64(anv_address_add(addr, 0)));
}

static void
copy_query_results_with_cs(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_query_pool *pool,
                           struct anv_address dest_addr,
                           uint64_t dest_stride,
                           uint32_t first_query,
                           uint32_t query_count,
                           VkQueryResultFlags flags)
{
   enum anv_pipe_bits needed_flushes =
      cmd_buffer->state.queries.buffer_write_bits |
      cmd_buffer->state.queries.clear_bits;

   trace_intel_begin_query_copy_cs(&cmd_buffer->trace);

   /* Occlusion & timestamp queries are written using a PIPE_CONTROL and
    * because we're about to copy values from MI commands, we need to stall
    * the command streamer to make sure the PIPE_CONTROL values have
    * landed, otherwise we could see inconsistent values & availability.
    *
    *  From the vulkan spec:
    *
    *     "vkCmdCopyQueryPoolResults is guaranteed to see the effect of
    *     previous uses of vkCmdResetQueryPool in the same queue, without any
    *     additional synchronization."
    */
   if (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION ||
       pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP)
      needed_flushes |= ANV_PIPE_CS_STALL_BIT;

   if (needed_flushes) {
      anv_add_pending_pipe_bits(cmd_buffer,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                needed_flushes,
                                "CopyQueryPoolResults");
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);
   }


#if GFX_VER >= 20
   /* On Gfx20+, if we're waiting for the results but there is no CS stall, we
    * need to fence the MI write emitted by previous vkCmdQuery(Begin|End)*
    * commands so the results are visible by MI read commands.
    */
   if (needed_flushes == 0 && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_MEM_FENCE), fence) {
         fence.FenceType = FENCE_TYPE_MI_WRITE;
      }
   }
#endif

   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);
   mi_builder_set_mocs(&b, anv_mocs_for_address(
                          cmd_buffer->device,
                          &(struct anv_address) { .bo = pool->bo }));

   for (uint32_t i = 0; i < query_count; i++) {
      struct anv_address query_addr = anv_query_address(pool, first_query + i);
      struct mi_value result;

      /* Wait for the availability write to land before we go read the data */
      if (flags & VK_QUERY_RESULT_WAIT_BIT) {
         anv_batch_emit(&cmd_buffer->batch, GENX(MI_SEMAPHORE_WAIT), sem) {
            sem.WaitMode            = PollingMode;
            sem.CompareOperation    = COMPARE_SAD_EQUAL_SDD;
            sem.SemaphoreDataDword  = true;
            sem.SemaphoreAddress    = query_addr;
         }
      }

      if (flags & VK_QUERY_RESULT_PARTIAL_BIT)
         mi_store(&b, mi_reg64(MI_PREDICATE_SRC0), mi_mem64(query_addr));

      uint32_t idx = 0;
      switch (pool->vk.query_type) {
      case VK_QUERY_TYPE_OCCLUSION:
      case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
#if GFX_VERx10 >= 125
      case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
         result = compute_query_result(&b, anv_address_add(query_addr, 8));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         break;

      case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
         uint32_t statistics = pool->vk.pipeline_statistics;
         while (statistics) {
            UNUSED uint32_t stat = u_bit_scan(&statistics);
            result = compute_query_result(&b, anv_address_add(query_addr,
                                                              idx * 16 + 8));
            gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                   flags, idx++, result);
         }
         assert(idx == util_bitcount(pool->vk.pipeline_statistics));
         break;
      }

      case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
         result = compute_query_result(&b, anv_address_add(query_addr, 8));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         result = compute_query_result(&b, anv_address_add(query_addr, 24));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         break;

      case VK_QUERY_TYPE_TIMESTAMP:
         result = mi_mem64(anv_address_add(query_addr, 8));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         break;

#if GFX_VERx10 >= 125
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
         result = mi_mem64(anv_address_add(query_addr, 8));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         break;

      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
         result = mi_mem64(anv_address_add(query_addr, 16));
         gpu_write_query_result(cmd_buffer, &b, query_addr, dest_addr,
                                flags, idx++, result);
         break;
#endif

      case VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR:
         UNREACHABLE("Copy KHR performance query results not implemented");
         break;

      default:
         UNREACHABLE("unhandled query type");
      }

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         gpu_write_query_result(cmd_buffer, &b,
                                ANV_NULL_ADDRESS, dest_addr,
                                flags & ~VK_QUERY_RESULT_PARTIAL_BIT,
                                idx,
                                mi_mem64(query_addr));
      }

      dest_addr = anv_address_add(dest_addr, dest_stride);
   }

   trace_intel_end_query_copy_cs(&cmd_buffer->trace, query_count);
}

static void
copy_query_results_with_shader(struct anv_cmd_buffer *cmd_buffer,
                               struct anv_query_pool *pool,
                               struct anv_address dest_addr,
                               uint64_t dest_stride,
                               uint32_t first_query,
                               uint32_t query_count,
                               VkQueryResultFlags flags)
{
   VkPipelineStageFlags2 wait_stages = 0;
   enum anv_pipe_bits needed_flushes = 0;

   trace_intel_begin_query_copy_shader(&cmd_buffer->trace);

   /* Ensure all query MI writes are visible to the shader */
   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);
   mi_ensure_write_fence(&b);

   /* If this is the first command in the batch buffer, make sure we have
    * consistent pipeline mode.
    */
   if (cmd_buffer->state.current_pipeline == UINT32_MAX) {
      if (anv_cmd_buffer_blorp_uses_compute(cmd_buffer))
         genX(flush_pipeline_select_gpgpu)(cmd_buffer, false);
      else
         genX(flush_pipeline_select_3d)(cmd_buffer);
   }

#if GFX_VER >= 20
   /* On Gfx20+ there is no pipeline switching cost and we run everything on the 3D engine */
   if (anv_cmd_buffer_is_render_queue(cmd_buffer))
      genX(flush_pipeline_select_3d)(cmd_buffer);
#endif

   if ((cmd_buffer->state.queries.buffer_write_bits |
        cmd_buffer->state.queries.clear_bits) & ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT)
      wait_stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   if ((cmd_buffer->state.queries.buffer_write_bits |
        cmd_buffer->state.queries.clear_bits) & (ANV_PIPE_UNTYPED_DATAPORT_CACHE_FLUSH_BIT |
                                                 ANV_PIPE_HDC_PIPELINE_FLUSH_BIT |
                                                 ANV_PIPE_DATA_CACHE_FLUSH_BIT))
      wait_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

   needed_flushes |= cmd_buffer->state.queries.buffer_write_bits |
                     cmd_buffer->state.queries.clear_bits;

   /* Flushes for the queries to complete */
   if (flags & VK_QUERY_RESULT_WAIT_BIT) {
      /* We need to stall for previous CS writes to land or the flushes to
       * complete.
       */
      needed_flushes |= ANV_PIPE_CS_STALL_BIT;
   }

   /* Occlusion & timestamp queries are written using a PIPE_CONTROL and
    * because we're about to copy values from MI commands, we need to stall
    * the command streamer to make sure the PIPE_CONTROL values have
    * landed, otherwise we could see inconsistent values & availability.
    *
    *  From the vulkan spec:
    *
    *     "vkCmdCopyQueryPoolResults is guaranteed to see the effect of
    *     previous uses of vkCmdResetQueryPool in the same queue, without any
    *     additional synchronization."
    */
   if (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION ||
       pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP)
      needed_flushes |= ANV_PIPE_CS_STALL_BIT;

   if (needed_flushes) {
      anv_add_pending_pipe_bits(cmd_buffer,
                                wait_stages,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                needed_flushes | ANV_PIPE_END_OF_PIPE_SYNC_BIT,
                                "CopyQueryPoolResults");
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);
   }

   struct anv_shader_internal *copy_kernel;
   VkResult ret =
      anv_device_get_internal_shader(
         cmd_buffer->device,
         anv_cmd_buffer_blorp_uses_compute(cmd_buffer) ?
         ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_COMPUTE :
         ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_FRAGMENT,
         &copy_kernel);
   if (ret != VK_SUCCESS) {
      anv_batch_set_error(&cmd_buffer->batch, ret);
      return;
   }

   struct anv_simple_shader state = {
      .device               = cmd_buffer->device,
      .cmd_buffer           = cmd_buffer,
      .dynamic_state_stream = &cmd_buffer->dynamic_state_stream,
      .batch                = &cmd_buffer->batch,
      .kernel               = copy_kernel,
   };
   genX(emit_simple_shader_init)(&state);

   struct anv_state push_data_state =
      genX(simple_shader_alloc_push)(&state,
                                     sizeof(struct anv_query_copy_params));
   if (push_data_state.map == NULL)
      return;

   struct anv_query_copy_params *params = push_data_state.map;

   uint32_t copy_flags =
      ((flags & VK_QUERY_RESULT_64_BIT) ? ANV_COPY_QUERY_FLAG_RESULT64 : 0) |
      ((flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ? ANV_COPY_QUERY_FLAG_AVAILABLE : 0) |
      ((flags & VK_QUERY_RESULT_PARTIAL_BIT) ? ANV_COPY_QUERY_FLAG_PARTIAL : 0);

   uint32_t num_items = 1;
   uint32_t data_offset = 8 /* behind availability */;
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      copy_flags |= ANV_COPY_QUERY_FLAG_DELTA;
      break;

   case VK_QUERY_TYPE_TIMESTAMP:
      break;

   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
#if GFX_VERx10 >= 125
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
#endif
      copy_flags |= ANV_COPY_QUERY_FLAG_DELTA;
      break;

   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      num_items = util_bitcount(pool->vk.pipeline_statistics);
      copy_flags |= ANV_COPY_QUERY_FLAG_DELTA;
      break;

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      num_items = 2;
      copy_flags |= ANV_COPY_QUERY_FLAG_DELTA;
      break;

   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
      break;

   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      data_offset += 8;
      break;

   default:
      UNREACHABLE("unhandled query type");
   }

   *params = (struct anv_query_copy_params) {
      .flags              = copy_flags,
      .num_queries        = query_count,
      .num_items          = num_items,
      .query_base         = first_query,
      .query_stride       = pool->stride,
      .query_data_offset  = data_offset,
      .destination_stride = dest_stride,
      .query_data_addr    = anv_address_physical(
         (struct anv_address) {
            .bo = pool->bo,
         }),
      .destination_addr   = anv_address_physical(dest_addr),
   };

   genX(emit_simple_shader_dispatch)(&state, query_count, push_data_state);

   /* The query copy result shader is writing using the dataport, flush
    * HDC/Data cache depending on the generation. Also stall at pixel
    * scoreboard in case we're doing the copy with a fragment shader.
    */
   cmd_buffer->state.queries.copy_bits |=
      anv_cmd_buffer_shader_query_sync_bits(cmd_buffer);

   trace_intel_end_query_copy_shader(&cmd_buffer->trace, query_count);
}

void genX(CmdCopyQueryPoolResultsToMemoryKHR)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    const VkStridedDeviceAddressRangeKHR*       pDstRange,
    VkAddressCommandFlagsKHR                    dstFlags,
    VkQueryResultFlags                          queryResultFlags)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_device *device = cmd_buffer->device;
   const struct anv_physical_device *pdevice = device->physical;

   struct anv_address dst_addr =
      anv_address_from_strided_range_flags(*pDstRange, dstFlags);

   if (queryCount > pdevice->drirc.perf.query_copy_with_shader_threshold) {
      copy_query_results_with_shader(cmd_buffer, pool,
                                     dst_addr,
                                     pDstRange->stride,
                                     firstQuery,
                                     queryCount,
                                     queryResultFlags);
   } else {
      copy_query_results_with_cs(cmd_buffer, pool,
                                 dst_addr,
                                 pDstRange->stride,
                                 firstQuery,
                                 queryCount,
                                 queryResultFlags);
   }

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_TRANSFER;
}

#if GFX_VERx10 >= 125 && ANV_SUPPORT_RT

#include "bvh/anv_bvh_defines.h"

void
genX(CmdWriteAccelerationStructuresPropertiesKHR)(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery)
{
   assert(queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ||
          queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ||
          queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ||
          queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR);

   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);

   /* We need a CS stall for the flushing to complete before we run the MI
    * commands.
    *
    * If CS is also non coherent in L3, we need to flush L3.
    */
   anv_add_pending_pipe_bits(cmd_buffer,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             (ANV_DEVINFO_HAS_COHERENT_L3_CS(cmd_buffer->device->info) ? 0 :
                              ANV_PIPE_DATA_CACHE_FLUSH_BIT) |
                             ANV_PIPE_CS_STALL_BIT,
                             "read BVH data using CS");
   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   struct mi_builder b;
   mi_builder_init(&b, cmd_buffer->device->info, &cmd_buffer->batch);

   for (uint32_t i = 0; i < accelerationStructureCount; i++) {
      ANV_FROM_HANDLE(vk_acceleration_structure, accel, pAccelerationStructures[i]);
      struct anv_address query_addr =
         anv_address_add(anv_query_address(pool, firstQuery + i), 8);
      uint64_t va = vk_acceleration_structure_get_va(accel);

      mi_builder_set_write_check(&b, (i == (accelerationStructureCount - 1)));

      switch (queryType) {
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
         va += offsetof(struct anv_accel_struct_header, compacted_size);
         break;
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
         va += offsetof(struct anv_accel_struct_header, size);
         break;
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
         va += offsetof(struct anv_accel_struct_header, serialization_size);
         break;
      case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
         va += offsetof(struct anv_accel_struct_header, instance_count);
         /* To respect current set up tailored for GRL, the numBlasPtrs are
          * stored at the second slot (third slot, if you count availability)
          */
         query_addr = anv_address_add(query_addr, 8);
         break;
      default:
         UNREACHABLE("unhandled query type");
      }

      mi_store(&b, mi_mem64(query_addr), mi_mem64(anv_address_from_u64(va)));
   }

   struct mi_builder b1;
   mi_builder_init(&b1, cmd_buffer->device->info, &cmd_buffer->batch);

   for (uint32_t i = 0; i < accelerationStructureCount; i++) {
      mi_builder_set_write_check(&b1, (i == (accelerationStructureCount - 1)));
      emit_query_mi_availability(&b1, anv_query_address(pool, firstQuery + i), true);
   }

   cmd_buffer->state.last_cmd_type = ANV_CMD_TYPE_QUERY;
}
#endif /* GFX_VERx10 >= 125 && ANV_SUPPORT_RT */
