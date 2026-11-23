/*
 * Copyright © 2018 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "anv_private.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"

#include "perf/intel_perf.h"
#include "perf/intel_perf_mdapi.h"
#include "perf/intel_perf_metrics_library.h"

#include "util/mesa-blake3.h"

void
anv_physical_device_init_perf(struct anv_physical_device *device, int fd)
{
   struct intel_perf_config *perf = intel_perf_new(NULL);

   intel_perf_init_metrics(perf, &device->info, fd,
                           false /* pipeline statistics */,
                           true /* register snapshots */);

   if (!perf->n_queries)
      goto err;

   /* We need DRM_I915_PERF_PROP_HOLD_PREEMPTION support, only available in
    * perf revision 2.
    */
   if (!INTEL_DEBUG(DEBUG_NO_OACONFIG)) {
      if (!intel_perf_has_hold_preemption(perf))
         goto err;
   }

   device->perf = perf;

   /* Global OAG mode for KHR performance queries. The context-relative (OAR)
    * path MI_RPC uses cannot report GT-wide counters at all: they always read
    * back as zero. The uapi this needs (global stream, OA buffer mapping,
    * whitelisted OAG registers) only exists in the xe KMD, and the resolve
    * code relies on the Xe2+ report layout and on the kernel making the OA
    * buffer wrap at report boundaries, which it only does on Xe2+, so i915
    * and pre-Xe2 parts keep the OAR path. So does INTEL_DEBUG=no-oaconfig:
    * it never opens a stream, and without one the boundary reports this mode
    * resolves against are never written.
    */
   perf->oag_global_enable =
      device->info.kmd_type == INTEL_KMD_TYPE_XE &&
      device->info.verx10 >= 200 &&
      !INTEL_DEBUG(DEBUG_NO_OACONFIG);

   /* Compute the number of commands we need to implement a performance
    * query.
    */
   const struct intel_perf_query_field_layout *layout = &perf->query_layout;
   device->n_perf_query_commands = 0;
   for (uint32_t f = 0; f < layout->n_fields; f++) {
      struct intel_perf_query_field *field = &layout->fields[f];

      switch (field->type) {
      case INTEL_PERF_QUERY_FIELD_TYPE_MI_RPC:
         /* In OAG mode MI_RPC is replaced by a trigger write, which needs no
          * relocation, plus ANV_OAG_BOUNDARY_STORES stores that do.
          */
         device->n_perf_query_commands +=
            perf->oag_global_enable ? ANV_OAG_BOUNDARY_STORES : 1;
         break;
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_PERFCNT:
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_RPSTAT:
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_A:
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_B:
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_C:
      case INTEL_PERF_QUERY_FIELD_TYPE_SRM_OA_PEC:
         /* These read the context-relative OAR sub-unit, which is always zero
          * for GT-wide counters, so OAG mode does not emit them at all.
          */
         if (!perf->oag_global_enable)
            device->n_perf_query_commands += field->size / 4;
         break;
      default:
         UNREACHABLE("Unhandled register type");
      }
   }
   device->n_perf_query_commands *= 2; /* Begin & End */
   device->n_perf_query_commands += 1; /* availability */

   return;

err:
   intel_perf_free(perf);
}

void
anv_device_perf_init(struct anv_device *device)
{
   device->perf_fd = -1;
   device->perf_queue = NULL;
   simple_mtx_init(&device->perf_oag.mutex, mtx_plain);
   device->perf_oag.next_query_id = ANV_OAG_QUERY_ID_FIRST;
   list_inithead(&device->perf_oag.pools);
}

void
anv_device_perf_finish(struct anv_device *device)
{
   anv_device_perf_close(device);
   simple_mtx_destroy(&device->perf_oag.mutex);
}

/* Reserve the OAG MMIO trigger markers of a performance query pool.
 *
 * Markers are handed out monotonically from a high range so that a report left
 * behind in the OA buffer by a destroyed pool can never be mistaken for a live
 * one. The range is bounded, so exhausting it fails deterministically at pool
 * creation instead of letting a later query wait for a report that can never
 * be identified.
 */
VkResult
anv_oag_alloc_query_ids(struct anv_device *device, struct anv_query_pool *pool)
{
   const uint64_t id_count =
      (uint64_t)pool->vk.query_count * pool->n_passes * 2;
   VkResult result = VK_SUCCESS;

   if (id_count == 0 || id_count > UINT32_MAX - ANV_OAG_QUERY_ID_FIRST)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_lock(&device->perf_oag.mutex);
   if (device->perf_oag.next_query_id > UINT32_MAX - id_count) {
      result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   } else {
      pool->oag_query_id_base = device->perf_oag.next_query_id;
      device->perf_oag.next_query_id += id_count;
      device->perf_oag.n_query_pools++;
      list_addtail(&pool->oag_link, &device->perf_oag.pools);
   }
   simple_mtx_unlock(&device->perf_oag.mutex);

   return result;
}

/* Release the marker range of a performance query pool.
 *
 * Individual ranges are not recycled: a marker must stay unique for as long as
 * any report carrying it can still be sitting in the OA buffer, and there is
 * no cheap way to know that. The whole range is instead reclaimed at once when
 * the last performance query pool of the device goes away, which bounds the
 * marker space for a process that creates and destroys pools in a loop.
 */
void
anv_oag_free_query_ids(struct anv_device *device, struct anv_query_pool *pool)
{
   if (pool->oag_query_id_base == 0)
      return;

   simple_mtx_lock(&device->perf_oag.mutex);
   list_del(&pool->oag_link);
   assert(device->perf_oag.n_query_pools > 0);
   if (--device->perf_oag.n_query_pools == 0)
      device->perf_oag.next_query_id = ANV_OAG_QUERY_ID_FIRST;
   simple_mtx_unlock(&device->perf_oag.mutex);

   pool->oag_query_id_base = 0;
}

/* OAG address registers keep a 64B aligned address in their top bits. */
#define ANV_OAG_ADDRESS_MASK             0xffffffc0u

/* Find the report triggered by @marker within the OATAIL window recorded in
 * @boundary and copy it to @snapshot. Must be called with
 * device->perf_oag.mutex held.
 */
bool
anv_oag_resolve_boundary(struct anv_device *device,
                         struct anv_oag_boundary *boundary,
                         void *snapshot, uint32_t marker)
{
   /* Already copied out by an earlier vkGetQueryPoolResults(). The OA buffer
    * is a ring, by now the report itself may be long gone.
    */
   if (boundary->resolved == ANV_OAG_RESOLVED_MAGIC)
      return true;

   const uint32_t report_size = device->physical->perf->oa_sample_size;

   /* No trigger fired, accumulate a zero report instead of whatever an
    * earlier execution left in the host snapshot.
    */
   if (boundary->resolved == ANV_OAG_ZEROED_MAGIC) {
      memset(snapshot, 0, report_size);
      boundary->resolved = ANV_OAG_RESOLVED_MAGIC;
      return true;
   }

   const uint8_t *oa_buffer = device->perf_oag.oa_buffer;
   const uint32_t circ_size = device->perf_oag.oa_buffer_circ_size;

   if (!oa_buffer || circ_size < report_size)
      return false;

   /* OATAIL and OABUFFER both hold 64B aligned GGTT addresses; the latter is
    * what turns the former into an offset into the mapped buffer.
    */
   const uint32_t base = boundary->oa_buffer & ANV_OAG_ADDRESS_MASK;
   uint32_t first = (boundary->tail_pre & ANV_OAG_ADDRESS_MASK) - base;
   uint32_t last = (boundary->tail_post & ANV_OAG_ADDRESS_MASK) - base;

   if (first >= circ_size || last >= circ_size)
      return false;

   /* Widen the window to whole reports: the tail may point into a report that
    * was only partially written when it was sampled.
    */
   first -= first % report_size;
   if (last % report_size)
      last = (last + report_size - (last % report_size)) % circ_size;

   /* One report of slack past the trailing tail, in case the store of
    * OATAIL raced the write of the triggered report itself.
    */
   const uint32_t window = (last + circ_size - first) % circ_size;
   const uint32_t n_reports = window / report_size + 1;

   for (uint32_t i = 0; i < n_reports; i++) {
      const uint32_t offset = (first + i * report_size) % circ_size;
      const uint8_t *report = oa_buffer + offset;

      if (!(intel_perf_report_reason((const uint32_t *)report) &
            INTEL_PERF_OA_REPORT_REASON_MMIO_TRIGGER))
         continue;

      if (intel_perf_report_marker(report) != marker)
         continue;

      memcpy(snapshot, report, report_size);

      /* This scan may run while the OA unit is still flushing this very
       * report out (that lag is why the resolve is retried at all), and there
       * is no completion flag: the header dwords the match was made on can be
       * visible before the trailing counters are. Re-read and compare; a
       * report that changed under the copy is treated as not written yet and
       * left for a later attempt. In OAG mode nothing else lives in the
       * snapshot, so the discarded copy clobbers nothing.
       */
      if (memcmp(snapshot, report, report_size) != 0)
         return false;

      boundary->resolved = ANV_OAG_RESOLVED_MAGIC;
      return true;
   }

   return false;
}

/* Latch the boundary reports of every live performance query pool out of the
 * OA buffer. Called with perf_oag.mutex held, right before the mapping goes
 * away: vkGetQueryPoolResults() stays legal after vkReleaseProfilingLockKHR(),
 * so whatever has been executed but not read back must be resolved now or it
 * never will be.
 */
static void
anv_oag_resolve_all_pools_locked(struct anv_device *device)
{
   list_for_each_entry(struct anv_query_pool, pool,
                       &device->perf_oag.pools, oag_link) {
      /* Registered at ID allocation time, which precedes the BO allocation:
       * a pool still being created has no snapshots to resolve.
       */
      if (!pool->bo)
         continue;

      for (uint32_t q = 0; q < pool->vk.query_count; q++) {
         for (uint32_t p = 0; p < pool->n_passes; p++) {
            for (uint32_t end = 0; end < 2; end++) {
               struct anv_oag_boundary *boundary = pool->bo->map +
                  khr_perf_query_data_offset(pool, q, p, end);
               void *snapshot = pool->oag_snapshots +
                  khr_perf_query_snapshot_offset(pool, q, p, end);
               anv_oag_resolve_boundary(device, boundary, snapshot,
                                        anv_oag_query_id(pool, q, p, end));
            }
         }
      }
   }
}

void
anv_device_perf_close(struct anv_device *device)
{
   simple_mtx_lock(&device->perf_oag.mutex);
   if (device->perf_oag.oa_buffer) {
      anv_oag_resolve_all_pools_locked(device);
      intel_perf_stream_unmap_oa_buffer(device->perf_oag.oa_buffer,
                                        device->perf_oag.oa_buffer_size);
      device->perf_oag.oa_buffer = NULL;
      device->perf_oag.oa_buffer_size = 0;
      device->perf_oag.oa_buffer_circ_size = 0;
   }
   simple_mtx_unlock(&device->perf_oag.mutex);

   if (device->perf_fd != -1) {
      if (intel_bind_timeline_get_syncobj(&device->perf_timeline))
         intel_bind_timeline_finish(&device->perf_timeline, device->fd);
      close(device->perf_fd);
      device->perf_fd = -1;
   }
   device->perf_queue = NULL;
}

void
anv_oag_resolve_all_pools(struct anv_device *device)
{
   simple_mtx_lock(&device->perf_oag.mutex);
   if (device->perf_oag.oa_buffer)
      anv_oag_resolve_all_pools_locked(device);
   simple_mtx_unlock(&device->perf_oag.mutex);
}

/* Resolving only at vkGetQueryPoolResults() time lets a long replay wrap the
 * OA ring and overwrite boundary reports before anyone has looked at them
 * (e.g. RenderDoc replays a frame once per counter pass and fetches every
 * result at the end). Latch reports out at wait-idle instead: every submitted
 * query has executed by then, so the ring only has to hold one wait-idle
 * interval's worth of reports. vkDeviceWaitIdle() lands here too, the runtime
 * implements it with QueueWaitIdle on every queue.
 */
VkResult
anv_QueueWaitIdle(VkQueue _queue)
{
   ANV_FROM_HANDLE(anv_queue, queue, _queue);
   struct anv_device *device = queue->device;
   VkResult result = vk_common_QueueWaitIdle(_queue);

   if (result == VK_SUCCESS && device->physical->perf &&
       device->physical->perf->oag_global_enable)
      anv_oag_resolve_all_pools(device);

   return result;
}

static uint32_t
anv_device_perf_get_queue_context_or_exec_queue_id(struct anv_queue *queue)
{
   struct anv_device *device = queue->device;
   uint32_t context_or_exec_queue_id;

   switch (device->physical->info.kmd_type) {
   case INTEL_KMD_TYPE_I915:
      context_or_exec_queue_id = device->physical->has_vm_control ?
                                 queue->context_id : device->context_id;
      break;
   case INTEL_KMD_TYPE_XE:
      context_or_exec_queue_id = queue->exec_queue_id;
      break;
   default:
      UNREACHABLE("missing");
      context_or_exec_queue_id = 0;
   }

   return context_or_exec_queue_id;
}

static int
anv_device_perf_open(struct anv_device *device, struct anv_queue *queue,
                     uint64_t metric_id)
{
   uint64_t period_exponent = 31; /* slowest sampling period */
   uint32_t exec_id =
      anv_device_perf_get_queue_context_or_exec_queue_id(queue);
   int ret;

   /* Binding the OA stream to the submitting context/exec-queue programs the
    * context-scoped OAR sub-unit, whose GT-wide counters always read zero. In
    * OAG mode we open the global OAG unit instead. Sampling must stay enabled
    * for the OA buffer to exist and for the OAG registers to be whitelisted in
    * userspace batches, but no periodic report is ever consumed, so the
    * slowest period stays: periodic traffic only steals OA buffer space from
    * the boundary reports and widens the window they are searched in.
    */
   const bool use_global_oag = device->physical->perf->oag_global_enable;
   if (use_global_oag)
      exec_id = 0;

   if (intel_perf_has_metric_sync(device->physical->perf)) {
      if (!intel_bind_timeline_init(&device->perf_timeline, device->fd))
         return -1;
   }

   ret = intel_perf_stream_open(device->physical->perf, device->fd,
                                exec_id,
                                metric_id, period_exponent, true, true,
                                &device->perf_timeline);
   if (ret < 0)
      goto err_timeline;

   if (use_global_oag) {
      /* Boundary reports are read back straight out of the OA buffer, so a
       * mapping is mandatory in OAG mode. It is gated by the same
       * xe_observation_paranoid check that already allowed the stream open
       * above, so this is not expected to fail.
       */
      uint64_t map_size = 0;
      void *map = intel_perf_stream_map_oa_buffer(device->physical->perf,
                                                  ret, &map_size);
      if (!map)
         goto err_stream;

      simple_mtx_lock(&device->perf_oag.mutex);
      device->perf_oag.oa_buffer = map;
      device->perf_oag.oa_buffer_size = map_size;
      /* The kernel truncates the ring to a whole number of reports, so it is
       * generally not a power of two and wrapping has to go through this
       * value rather than a mask.
       */
      device->perf_oag.oa_buffer_circ_size =
         map_size - (map_size % device->physical->perf->oa_sample_size);
      simple_mtx_unlock(&device->perf_oag.mutex);
   }

   device->perf_queue = queue;

   return ret;

 err_stream:
   close(ret);
 err_timeline:
   intel_bind_timeline_finish(&device->perf_timeline, device->fd);

   return -1;
}

/* VK_INTEL_performance_query */
VkResult anv_InitializePerformanceApiINTEL(
    VkDevice                                    _device,
    const VkInitializePerformanceApiInfoINTEL*  pInitializeInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   if (!device->physical->perf)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

   if (!intel_perf_init_metrics_library(device->physical->perf, device->fd)) {
      /* Do not use Metrics Library if it fails to initialize */
      device->physical->perf->use_metrics_library = false;
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   /* Not much to do here */
   return VK_SUCCESS;
}

VkResult anv_GetPerformanceParameterINTEL(
    VkDevice                                    _device,
    VkPerformanceParameterTypeINTEL             parameter,
    VkPerformanceValueINTEL*                    pValue)
{
      ANV_FROM_HANDLE(anv_device, device, _device);

      if (!device->physical->perf)
         return VK_ERROR_EXTENSION_NOT_PRESENT;

      VkResult result = VK_SUCCESS;
      switch (parameter) {
      case VK_PERFORMANCE_PARAMETER_TYPE_HW_COUNTERS_SUPPORTED_INTEL:
         pValue->type = VK_PERFORMANCE_VALUE_TYPE_BOOL_INTEL;
         pValue->data.valueBool = VK_TRUE;
         break;

      case VK_PERFORMANCE_PARAMETER_TYPE_STREAM_MARKER_VALID_BITS_INTEL:
         pValue->type = VK_PERFORMANCE_VALUE_TYPE_UINT32_INTEL;
         pValue->data.value32 = 25;
         break;

      default:
         result = VK_ERROR_FEATURE_NOT_PRESENT;
         break;
      }

      return result;
}

VkResult anv_CmdSetPerformanceMarkerINTEL(
    VkCommandBuffer                             commandBuffer,
    const VkPerformanceMarkerInfoINTEL*         pMarkerInfo)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   cmd_buffer->intel_perf_marker = pMarkerInfo->marker;

   return VK_SUCCESS;
}

VkResult anv_AcquirePerformanceConfigurationINTEL(
    VkDevice                                    _device,
    const VkPerformanceConfigurationAcquireInfoINTEL* pAcquireInfo,
    VkPerformanceConfigurationINTEL*            pConfiguration)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_performance_configuration_intel *config;

   config = vk_object_alloc(&device->vk, NULL, sizeof(*config),
                            VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL);
   if (!config)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   config->config_id = intel_perf_metrics_library_create_configuration(device->physical->perf);

   if (config->config_id == 0) {
      vk_object_free(&device->vk, NULL, config);
      return VK_INCOMPLETE;
   }

   *pConfiguration = anv_performance_configuration_intel_to_handle(config);

   return VK_SUCCESS;
}

VkResult anv_ReleasePerformanceConfigurationINTEL(
    VkDevice                                    _device,
    VkPerformanceConfigurationINTEL             _configuration)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_performance_configuration_intel, config, _configuration);

   if (!intel_perf_metrics_library_destroy_configuration(device->physical->perf, config->config_id))
      vk_error(device, VK_ERROR_UNKNOWN);

   vk_object_free(&device->vk, NULL, config);

   return VK_SUCCESS;
}

static struct anv_queue *
anv_device_get_perf_queue(struct anv_device *device)
{
   for (uint32_t i = 0; i < device->queue_count; i++) {
      struct anv_queue *queue = &device->queues[i];
      const struct anv_queue_family *family = queue->family;

      if (family->supports_perf)
         return queue;
   }

   return NULL;
}

VkResult anv_QueueSetPerformanceConfigurationINTEL(
    VkQueue                                     _queue,
    VkPerformanceConfigurationINTEL             _configuration)
{
   ANV_FROM_HANDLE(anv_queue, queue, _queue);
   ANV_FROM_HANDLE(anv_performance_configuration_intel, config, _configuration);

   vk_queue_lock(&queue->vk);

   queue->metrics_library_configuration = config->config_id;

   vk_queue_unlock(&queue->vk);

   return VK_SUCCESS;
}

void anv_UninitializePerformanceApiINTEL(
    VkDevice                                    _device)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   intel_perf_deinit_metrics_library(device->physical->perf);

   anv_device_perf_close(device);
}

/* VK_KHR_performance_query */
static const VkPerformanceCounterUnitKHR
intel_perf_counter_unit_to_vk_unit[] = {
   [INTEL_PERF_COUNTER_UNITS_BYTES]                                = VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR,
   [INTEL_PERF_COUNTER_UNITS_HZ]                                   = VK_PERFORMANCE_COUNTER_UNIT_HERTZ_KHR,
   [INTEL_PERF_COUNTER_UNITS_NS]                                   = VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR,
   [INTEL_PERF_COUNTER_UNITS_US]                                   = VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR, /* todo */
   [INTEL_PERF_COUNTER_UNITS_PIXELS]                               = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_TEXELS]                               = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_THREADS]                              = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_PERCENT]                              = VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR,
   [INTEL_PERF_COUNTER_UNITS_MESSAGES]                             = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_NUMBER]                               = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_CYCLES]                               = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_EVENTS]                               = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_UTILIZATION]                          = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_EU_SENDS_TO_L3_CACHE_LINES]           = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_EU_ATOMIC_REQUESTS_TO_L3_CACHE_LINES] = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_EU_REQUESTS_TO_L3_CACHE_LINES]        = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
   [INTEL_PERF_COUNTER_UNITS_EU_BYTES_PER_L3_CACHE_LINE]           = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
};

static const VkPerformanceCounterStorageKHR
intel_perf_counter_data_type_to_vk_storage[] = {
   [INTEL_PERF_COUNTER_DATA_TYPE_BOOL32] = VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR,
   [INTEL_PERF_COUNTER_DATA_TYPE_UINT32] = VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR,
   [INTEL_PERF_COUNTER_DATA_TYPE_UINT64] = VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR,
   [INTEL_PERF_COUNTER_DATA_TYPE_FLOAT]  = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR,
   [INTEL_PERF_COUNTER_DATA_TYPE_DOUBLE] = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR,
};

VkResult anv_EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    uint32_t*                                   pCounterCount,
    VkPerformanceCounterKHR*                    pCounters,
    VkPerformanceCounterDescriptionKHR*         pCounterDescriptions)
{
   ANV_FROM_HANDLE(anv_physical_device, pdevice, physicalDevice);
   struct intel_perf_config *perf = pdevice->perf;

   uint32_t desc_count = *pCounterCount;

   VK_OUTARRAY_MAKE_TYPED(VkPerformanceCounterKHR, out, pCounters, pCounterCount);
   VK_OUTARRAY_MAKE_TYPED(VkPerformanceCounterDescriptionKHR, out_desc,
                          pCounterDescriptions, &desc_count);

   /* We cannot support performance queries on anything other than RCS,
    * because the MI_REPORT_PERF_COUNT command is not available on other
    * engines.
    */
   struct anv_queue_family *queue_family =
      &pdevice->queue.families[queueFamilyIndex];
   if (queue_family->engine_class != INTEL_ENGINE_CLASS_RENDER)
      return vk_outarray_status(&out);

   for (int c = 0; c < (perf ? perf->n_counters : 0); c++) {
      const struct intel_perf_query_counter *intel_counter = perf->counter_infos[c].counter;

      vk_outarray_append_typed(VkPerformanceCounterKHR, &out, counter) {
         counter->unit = intel_perf_counter_unit_to_vk_unit[intel_counter->units];
         counter->scope = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR;
         counter->storage = intel_perf_counter_data_type_to_vk_storage[intel_counter->data_type];

         unsigned char blake3_result[BLAKE3_KEY_LEN];
         _mesa_blake3_compute(intel_counter->symbol_name,
                            strlen(intel_counter->symbol_name),
                            blake3_result);
         memcpy(counter->uuid, blake3_result, sizeof(counter->uuid));
      }

      vk_outarray_append_typed(VkPerformanceCounterDescriptionKHR, &out_desc, desc) {
         desc->flags = pdevice->perf->oag_global_enable ?
            VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR : 0;
         snprintf(desc->name, sizeof(desc->name), "%s",
                  INTEL_DEBUG(DEBUG_PERF_SYMBOL_NAMES) ?
                  intel_counter->symbol_name :
                  intel_counter->name);
         snprintf(desc->category, sizeof(desc->category), "%s", intel_counter->category);
         snprintf(desc->description, sizeof(desc->description), "%s", intel_counter->desc);
      }
   }

   return vk_outarray_status(&out);
}

void anv_GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkQueryPoolPerformanceCreateInfoKHR*  pPerformanceQueryCreateInfo,
    uint32_t*                                   pNumPasses)
{
   ANV_FROM_HANDLE(anv_physical_device, pdevice, physicalDevice);
   struct intel_perf_config *perf = pdevice->perf;

   if (!perf) {
      *pNumPasses = 0;
      return;
   }

   *pNumPasses = intel_perf_get_n_passes(perf,
                                       pPerformanceQueryCreateInfo->pCounterIndices,
                                       pPerformanceQueryCreateInfo->counterIndexCount,
                                       NULL);
}

VkResult anv_AcquireProfilingLockKHR(
    VkDevice                                    _device,
    const VkAcquireProfilingLockInfoKHR*        pInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct intel_perf_config *perf = device->physical->perf;
   struct intel_perf_query_info *first_metric_set = &perf->queries[0];
   int fd = -1;

   if (device->perf_fd != -1)
      return VK_TIMEOUT;

   if (!INTEL_DEBUG(DEBUG_NO_OACONFIG)) {
      struct anv_queue *queue = anv_device_get_perf_queue(device);

      if (queue == NULL)
         return VK_ERROR_UNKNOWN;
      fd = anv_device_perf_open(device, queue,
                                first_metric_set->oa_metrics_set_id);
      if (fd < 0)
         return VK_TIMEOUT;
   }

   device->perf_fd = fd;
   return VK_SUCCESS;
}

void anv_ReleaseProfilingLockKHR(
    VkDevice                                    _device)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   anv_device_perf_close(device);
}

void
anv_perf_write_pass_results(struct intel_perf_config *perf,
                            struct anv_query_pool *pool, uint32_t pass,
                            const struct intel_perf_query_result *accumulated_results,
                            union VkPerformanceCounterResultKHR *results)
{
   const struct intel_perf_query_info *query = pool->pass_query[pass];

   for (uint32_t c = 0; c < pool->n_counters; c++) {
      const struct intel_perf_counter_pass *counter_pass = &pool->counter_pass[c];

      if (counter_pass->query != query)
         continue;

      switch (pool->pass_query[pass]->kind) {
      case INTEL_PERF_QUERY_TYPE_PIPELINE: {
         assert(counter_pass->counter->data_type == INTEL_PERF_COUNTER_DATA_TYPE_UINT64);
         uint32_t accu_offset = counter_pass->counter->offset / sizeof(uint64_t);
         results[c].uint64 = accumulated_results->accumulator[accu_offset];
         break;
      }

      case INTEL_PERF_QUERY_TYPE_OA:
      case INTEL_PERF_QUERY_TYPE_RAW:
         switch (counter_pass->counter->data_type) {
         case INTEL_PERF_COUNTER_DATA_TYPE_UINT64:
            results[c].uint64 =
               counter_pass->counter->oa_counter_read_uint64(perf,
                                                             counter_pass->query,
                                                             accumulated_results);
            break;
         case INTEL_PERF_COUNTER_DATA_TYPE_FLOAT:
            results[c].float32 =
               counter_pass->counter->oa_counter_read_float(perf,
                                                            counter_pass->query,
                                                            accumulated_results);
            break;
         default:
            /* So far we aren't using uint32, double or bool32... */
            UNREACHABLE("unexpected counter data type");
         }
         break;

      default:
         UNREACHABLE("invalid query type");
      }

      /* The Vulkan extension only has nanoseconds as a unit */
      if (counter_pass->counter->units == INTEL_PERF_COUNTER_UNITS_US) {
         assert(counter_pass->counter->data_type == INTEL_PERF_COUNTER_DATA_TYPE_UINT64);
         results[c].uint64 *= 1000;
      }
   }
}
