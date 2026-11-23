/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * 'pvr_write_query_to_buffer()' and 'pvr_wait_for_available()' based on anv:
 * Copyright © 2015 Intel Corporation
 */

#include "pvr_query.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "hwdef/pvr_hw_utils.h"
#include "pvr_bo.h"
#include "pvr_csb.h"
#include "pvr_device.h"
#include "pvr_device_info.h"
#include "pvr_entrypoints.h"
#include "pvr_hw_pass.h"
#include "pvr_macros.h"
#include "pvr_pass.h"
#include "pvr_physical_device.h"
#include "pvr_xfb.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "vk_log.h"
#include "vk_object.h"

static uint32_t pvr_reports_per_query(const struct pvr_query_pool *pool)
{
   switch (pool->query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      return 1u;

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      /* {prims written, prims needed} */
      return 2u;

   default:
      break;
   }

   UNREACHABLE("Unsupported query type");
}

static VkResult pvr_occlusion_query_pool_setup(struct pvr_device *device,
                                               struct pvr_query_pool *pool)
{
   const uint32_t core_count = device->pdevice->dev_runtime_info.core_count;
   uint64_t alloc_size;
   uint32_t result_size;

   /* Occlusion queries return 1 result per query. */
   pool->result_elem_stride = sizeof(uint32_t);

   result_size = pool->query_count * pool->result_elem_stride;
   pool->result_stride = ALIGN_POT(result_size, ROGUE_OQUERY_ALIGN);

   /* Each Phantom writes to a separate offset within the vis test heap so
    * allocate space for the total number of Phantoms.
    */
   alloc_size = (uint64_t)pool->result_stride * core_count;

   return pvr_bo_suballoc(&device->suballoc_vis_test,
                          alloc_size,
                          ROGUE_OQUERY_ALIGN,
                          false,
                          &pool->result_buffer);
}

static VkResult pvr_xfb_query_pool_setup(struct pvr_device *device,
                                         struct pvr_query_pool *pool)
{
   const uint32_t counter_size =
      pvr_xfb_get_counter_bytes(&device->pdevice->dev_info);
   const uint32_t cache_line_size =
      pvr_get_slc_cache_line_size(&device->pdevice->dev_info);
   const uint32_t elem_size =
      ALIGN_POT(pool->query_report_count * counter_size,
                PVR_XFB_QUERY_RESULT_ALIGN);
   const uint64_t result_buffer_alloc_size =
      (uint64_t)pool->query_count * elem_size;

   pool->result_elem_stride = elem_size;

   if (result_buffer_alloc_size > UINT32_MAX)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   return pvr_bo_suballoc(&device->suballoc_general,
                          result_buffer_alloc_size,
                          cache_line_size,
                          true,
                          &pool->result_buffer);
}

static VkResult pvr_query_pool_setup(struct pvr_device *device,
                                     struct pvr_query_pool *pool)
{
   /* Vulkan 1.0 supports only occlusion, timestamp, and pipeline statistics
    * query.
    * We don't currently support timestamp queries.
    * VkQueueFamilyProperties->timestampValidBits = 0.
    * We don't currently support pipeline statistics queries.
    * VkPhysicalDeviceFeatures->pipelineStatisticsQuery = false.
    */
   assert(!device->vk.enabled_features.pipelineStatisticsQuery);

   pool->query_report_count = pvr_reports_per_query(pool);

   switch (pool->query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      return pvr_occlusion_query_pool_setup(device, pool);

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      return pvr_xfb_query_pool_setup(device, pool);

   default:
      UNREACHABLE("Unsupported query type");
   }
}

static inline void
pvr_occlusion_query_pool_cleanup(const struct pvr_query_pool *pool)
{
   pvr_bo_suballoc_free(pool->result_buffer);
}

static inline void
pvr_xfb_query_pool_cleanup(const struct pvr_query_pool *pool)
{
   pvr_bo_suballoc_free(pool->result_buffer);
}

static inline void pvr_query_pool_cleanup(const struct pvr_query_pool *pool)
{
   if (!pool)
      return;

   switch (pool->query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
      pvr_occlusion_query_pool_cleanup(pool);
      break;

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      pvr_xfb_query_pool_cleanup(pool);
      break;

   default:
      break;
   }
}

VkResult pvr_CreateQueryPool(VkDevice _device,
                             const VkQueryPoolCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(pvr_device, device, _device);
   struct pvr_query_pool *pool;
   uint32_t query_size;
   VkResult result;

   pool = vk_object_alloc(&device->vk,
                          pAllocator,
                          sizeof(*pool),
                          VK_OBJECT_TYPE_QUERY_POOL);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->query_count = pCreateInfo->queryCount;
   pool->query_type = pCreateInfo->queryType;
   query_size = pCreateInfo->queryCount * sizeof(uint32_t);

   result = pvr_query_pool_setup(device, pool);
   if (result != VK_SUCCESS)
      goto err_free_pool;

   result = pvr_bo_suballoc(&device->suballoc_general,
                            query_size,
                            sizeof(uint32_t),
                            false,
                            &pool->availability_buffer);
   if (result != VK_SUCCESS)
      goto err_query_pool_cleanup;

   *pQueryPool = pvr_query_pool_to_handle(pool);

   return VK_SUCCESS;

err_query_pool_cleanup:
   pvr_query_pool_cleanup(pool);

err_free_pool:
   vk_object_free(&device->vk, pAllocator, pool);

   return result;
}

void pvr_DestroyQueryPool(VkDevice _device,
                          VkQueryPool queryPool,
                          const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(pvr_query_pool, pool, queryPool);
   VK_FROM_HANDLE(pvr_device, device, _device);

   if (!pool)
      return;

   pvr_bo_suballoc_free(pool->availability_buffer);
   pvr_query_pool_cleanup(pool);

   vk_object_free(&device->vk, pAllocator, pool);
}

/* Note: make sure to make the availability buffer's memory defined in
 * accordance to how the device is expected to fill it. We don't make it defined
 * here since that would cover up usage of this function while the underlying
 * buffer region being accessed wasn't expect to have been written by the
 * device.
 */
static inline bool pvr_query_is_available(const struct pvr_query_pool *pool,
                                          uint32_t query_idx)
{
   volatile uint32_t *available =
      pvr_bo_suballoc_get_map_addr(pool->availability_buffer);
   return !!available[query_idx];
}

#define NSEC_PER_SEC UINT64_C(1000000000)
#define PVR_WAIT_TIMEOUT UINT64_C(5)

/* Note: make sure to make the availability buffer's memory defined in
 * accordance to how the device is expected to fill it. We don't make it defined
 * here since that would cover up usage of this function while the underlying
 * buffer region being accessed wasn't expect to have been written by the
 * device.
 */
/* TODO: Handle device loss scenario properly. */
static VkResult pvr_wait_for_available(struct pvr_device *device,
                                       const struct pvr_query_pool *pool,
                                       uint32_t query_idx)
{
   const uint64_t abs_timeout =
      os_time_get_absolute_timeout(PVR_WAIT_TIMEOUT * NSEC_PER_SEC);

   /* From the Vulkan 1.0 spec:
    *
    *    Commands that wait indefinitely for device execution (namely
    *    vkDeviceWaitIdle, vkQueueWaitIdle, vkWaitForFences or
    *    vkAcquireNextImageKHR with a maximum timeout, and
    *    vkGetQueryPoolResults with the VK_QUERY_RESULT_WAIT_BIT bit set in
    *    flags) must return in finite time even in the case of a lost device,
    *    and return either VK_SUCCESS or VK_ERROR_DEVICE_LOST.
    */
   while (os_time_get_nano() < abs_timeout) {
      if (pvr_query_is_available(pool, query_idx) != 0)
         return VK_SUCCESS;
   }

   return vk_error(device, VK_ERROR_DEVICE_LOST);
}

#undef NSEC_PER_SEC
#undef PVR_WAIT_TIMEOUT

static inline void pvr_write_query_to_buffer(uint8_t *buffer,
                                             VkQueryResultFlags flags,
                                             uint32_t idx,
                                             uint64_t value)
{
   if (flags & VK_QUERY_RESULT_64_BIT) {
      uint64_t *query_data = (uint64_t *)buffer;
      query_data[idx] = value;
   } else {
      uint32_t *query_data = (uint32_t *)buffer;
      query_data[idx] = value;
   }
}

static void
pvr_occlusion_query_pool_get_result(struct pvr_device *device,
                                    const struct pvr_query_pool *pool,
                                    uint32_t query_idx,
                                    uint64_t *count_out)
{
   volatile uint32_t *query_results =
      pvr_bo_suballoc_get_map_addr(pool->result_buffer);
   const uint32_t core_count = device->pdevice->dev_runtime_info.core_count;
   uint64_t count = 0;

   for (uint32_t i = 0; i < core_count; i++)
      count += query_results[pool->result_stride * i + query_idx];

   *count_out = count;
}

static void
pvr_xfb_query_pool_get_result(struct pvr_device *device,
                              const struct pvr_query_pool *pool,
                              uint32_t query_idx,
                              uint64_t *result_out)
{
   const volatile uint32_t *counter_data =
      pvr_bo_suballoc_get_map_addr(pool->result_buffer);
   uint32_t offset =
      query_idx * pool->result_elem_stride / sizeof(*counter_data);
   uint64_t result = 0;

   for (uint32_t i = 0; i < pool->query_report_count; i++)
      result += counter_data[offset + i];

   *result_out = result;
}

VkResult pvr_GetQueryPoolResults(VkDevice _device,
                                 VkQueryPool queryPool,
                                 uint32_t firstQuery,
                                 uint32_t queryCount,
                                 size_t dataSize,
                                 void *pData,
                                 VkDeviceSize stride,
                                 VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(pvr_query_pool, pool, queryPool);
   VK_FROM_HANDLE(pvr_device, device, _device);
   VG(volatile uint32_t *available =
         pvr_bo_suballoc_get_map_addr(pool->availability_buffer));
   VG(volatile uint32_t *query_results =
      pvr_bo_suballoc_get_map_addr(pool->result_buffer));
   const uint32_t core_count = device->pdevice->dev_runtime_info.core_count;
   uint8_t *data = (uint8_t *)pData;
   VkResult result = VK_SUCCESS;

   /* TODO: Instead of making the memory defined here for valgrind, to better
    * catch out of bounds access and other memory errors we should move them
    * where where the query buffers are changed by the driver or device (e.g.
    * "vkCmdResetQueryPool()", "vkGetQueryPoolResults()", etc.).
    */

   VG(VALGRIND_MAKE_MEM_DEFINED(&available[firstQuery],
                                queryCount * sizeof(uint32_t)));

   for (uint32_t i = 0; i < core_count; i++) {
      VG(VALGRIND_MAKE_MEM_DEFINED(
         &query_results[firstQuery + i * pool->result_stride],
         queryCount * sizeof(uint32_t)));
   }

   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t query_idx = firstQuery + i;
      bool is_available = pvr_query_is_available(pool, query_idx);
      uint64_t count;

      if (flags & VK_QUERY_RESULT_WAIT_BIT && !is_available) {
         result = pvr_wait_for_available(device, pool, query_idx);
         if (result != VK_SUCCESS)
            return result;

         is_available = true;
      }

      switch (pool->query_type) {
      case VK_QUERY_TYPE_OCCLUSION:
         pvr_occlusion_query_pool_get_result(device, pool, query_idx, &count);
         break;

      case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
         pvr_xfb_query_pool_get_result(device, pool, query_idx, &count);
         break;

      default:
         UNREACHABLE("Unsupported query type");
      }

      if (is_available || (flags & VK_QUERY_RESULT_PARTIAL_BIT))
         pvr_write_query_to_buffer(data, flags, 0, count);
      else
         result = VK_NOT_READY;

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         pvr_write_query_to_buffer(data, flags, 1, is_available);

      data += stride;
   }

   VG(VALGRIND_MAKE_MEM_UNDEFINED(&available[firstQuery],
                                  queryCount * sizeof(uint32_t)));

   for (uint32_t i = 0; i < core_count; i++) {
      VG(VALGRIND_MAKE_MEM_UNDEFINED(
         &query_results[firstQuery + i * pool->result_stride],
         queryCount * sizeof(uint32_t)));
   }

   return result;
}

void pvr_ResetQueryPool(VkDevice _device,
                        VkQueryPool queryPool,
                        uint32_t firstQuery,
                        uint32_t queryCount)
{
   VK_FROM_HANDLE(pvr_query_pool, pool, queryPool);
   uint32_t *availability =
      pvr_bo_suballoc_get_map_addr(pool->availability_buffer);

   memset(availability + firstQuery, 0, sizeof(uint32_t) * queryCount);
}
