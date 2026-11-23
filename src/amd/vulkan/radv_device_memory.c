/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_device_memory.h"
#include "tools/radv_debug.h"
#include "tools/radv_rmv.h"
#include "radv_android.h"
#include "radv_buffer.h"
#include "radv_entrypoints.h"
#include "radv_image.h"

#include "vk_debug_utils.h"
#include "vk_log.h"

void
radv_free_memory(struct radv_device *device, const VkAllocationCallbacks *pAllocator, struct radv_device_memory *mem)
{
   if (mem == NULL)
      return;

   if (mem->bo) {
      radv_va_validation_update_page(device, radv_buffer_get_va(mem->bo), mem->alloc_size, false);

      if (device->overallocation_disallowed) {
         mtx_lock(&device->overallocation_mutex);
         device->allocated_memory_size[mem->heap_index] -= mem->alloc_size;
         mtx_unlock(&device->overallocation_mutex);
      }

      device->ws->buffer_make_resident(device->ws, mem->bo, false);
      radv_bo_destroy(device, &mem->vk.base, mem->bo);
      mem->bo = NULL;
   }

   radv_rmv_log_resource_destroy(device, (uint64_t)radv_device_memory_to_handle(mem));
   vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
}

VkResult
radv_alloc_memory(struct radv_device *device, const VkMemoryAllocateInfo *pAllocateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem, bool is_internal)
{
   struct radv_device_memory *mem;
   VkResult result;
   enum radeon_bo_domain domain;
   uint32_t flags = 0;

   assert(pAllocateInfo->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);

   const VkImportMemoryFdInfoKHR *import_info = vk_find_struct_const(pAllocateInfo->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   const VkMemoryDedicatedAllocateInfo *dedicate_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   const VkExportMemoryAllocateInfo *export_info =
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   const struct VkImportAndroidHardwareBufferInfoANDROID *ahb_import_info =
      vk_find_struct_const(pAllocateInfo->pNext, IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID);

   const struct wsi_memory_allocate_info *wsi_info =
      vk_find_struct_const(pAllocateInfo->pNext, WSI_MEMORY_ALLOCATE_INFO_MESA);

   if (pAllocateInfo->allocationSize == 0 && !ahb_import_info &&
       !(export_info &&
         (export_info->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID))) {
      /* Apparently, this is allowed */
      *pMem = VK_NULL_HANDLE;
      return VK_SUCCESS;
   }

   mem = vk_device_memory_create(&device->vk, pAllocateInfo, pAllocator, sizeof(*mem));
   if (mem == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (dedicate_info) {
      mem->image = radv_image_from_handle(dedicate_info->image);
      mem->buffer = radv_buffer_from_handle(dedicate_info->buffer);
   } else {
      mem->image = NULL;
      mem->buffer = NULL;
   }

   if (wsi_info) {
      /* Even if the WSI is managing implicit-sync fencing on the dma-buf with
       * dma-buf fence import/export, RADEON_FLAG_IMPLICIT_SYNC also controls
       * whether other users of the BO (such as an X server using an
       * implicit-syncing GL driver for rendering) will respect the BO's
       * implicit sync.
       */
      if (wsi_info->implicit_sync || wsi_info->dma_buf_sync_file)
         flags |= RADEON_FLAG_IMPLICIT_SYNC;

      /* Mark the linear prime buffer (aka the destination of the prime blit
       * as uncached.
       */
      if (mem->buffer)
         flags |= RADEON_FLAG_GL2_BYPASS;
   }

   float priority_float = 0.5;
   const struct VkMemoryPriorityAllocateInfoEXT *priority_ext =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_PRIORITY_ALLOCATE_INFO_EXT);
   if (priority_ext)
      priority_float = priority_ext->priority;

   uint64_t replay_address = 0;
   const VkMemoryOpaqueCaptureAddressAllocateInfo *replay_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);
   if (replay_info && replay_info->opaqueCaptureAddress)
      replay_address = replay_info->opaqueCaptureAddress;

   unsigned priority =
      MIN2(RADV_BO_PRIORITY_APPLICATION_MAX - 1, (int)(priority_float * RADV_BO_PRIORITY_APPLICATION_MAX));

   if (mem->vk.ahardware_buffer) {
      result = radv_import_ahb_memory(device, mem, priority);
      if (result != VK_SUCCESS)
         goto fail;
   } else if (import_info) {
      assert(import_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT ||
             import_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
      result = radv_bo_from_fd(device, import_info->fd, priority, mem, NULL);
      if (result != VK_SUCCESS) {
         goto fail;
      }
   } else if (mem->vk.host_ptr) {
      result = radv_bo_from_ptr(device, mem->vk.host_ptr, pAllocateInfo->allocationSize, priority, mem);
      if (result != VK_SUCCESS)
         goto fail;
   } else {
      const struct radv_physical_device *pdev = radv_device_physical(device);
      uint64_t alloc_size = align64(pAllocateInfo->allocationSize, 4096);
      uint32_t heap_index;

      heap_index = pdev->memory_properties.memoryTypes[pAllocateInfo->memoryTypeIndex].heapIndex;
      domain = pdev->memory_domains[pAllocateInfo->memoryTypeIndex];
      flags |= pdev->memory_flags[pAllocateInfo->memoryTypeIndex];

      if (mem->vk.export_handle_types) {
         /* Setting RADEON_FLAG_GTT_WC in case the bo is spilled to GTT.  This is important when the
          * foreign queue is the display engine of iGPU.  The carveout of iGPU can be tiny and the
          * kernel driver refuses to spill without the flag.
          *
          * This covers any external memory user, including WSI.
          */
         if (domain == RADEON_DOMAIN_VRAM)
            flags |= RADEON_FLAG_GTT_WC;
      } else if (!import_info) {
         /* neither export nor import */
         flags |= RADEON_FLAG_NO_INTERPROCESS_SHARING | RADEON_FLAG_PREFER_LOCAL_BO;
      }

      if (mem->vk.alloc_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT)
         flags |= RADEON_FLAG_REPLAYABLE;

      /* Only apply the workaround for BOs created by the application, not by the driver. */
      if (pdev->drirc.debug.wait_for_vm_map_updates)
         flags |= RADEON_FLAG_VM_UPDATE_WAIT;

      /* On GFX12, DCC is transparent to the userspace driver and PTE.DCC is
       * set per buffer allocation. Only VRAM can have DCC. When the kernel
       * moves a buffer from VRAM->GTT it decompresses. When the kernel moves
       * it from GTT->VRAM it recompresses but only if WRITE_COMPRESS_DISABLE=0
       * (see DCC tiling flags).
       */
      if (pdev->info.gfx_level >= GFX12 && pdev->info.gfx12_supports_dcc_write_compress_disable &&
          domain == RADEON_DOMAIN_VRAM && (flags & RADEON_FLAG_NO_CPU_ACCESS) && !radv_is_dcc_disabled(pdev) &&
          (!mem->image || mem->image->planes[0].surface.u.gfx9.gfx12_enable_dcc)) {
         flags |= RADEON_FLAG_GFX12_ALLOW_DCC;
      }

      if (device->overallocation_disallowed) {
         uint64_t total_size = pdev->memory_properties.memoryHeaps[heap_index].size;

         mtx_lock(&device->overallocation_mutex);
         if (device->allocated_memory_size[heap_index] + alloc_size > total_size) {
            mtx_unlock(&device->overallocation_mutex);
            result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            goto fail;
         }
         device->allocated_memory_size[heap_index] += alloc_size;
         mtx_unlock(&device->overallocation_mutex);
      }

      result = radv_bo_create(device, &mem->vk.base, alloc_size, pdev->info.max_alignment, domain, flags, priority,
                              replay_address, is_internal, &mem->bo);

      if (result != VK_SUCCESS) {
         if (device->overallocation_disallowed) {
            mtx_lock(&device->overallocation_mutex);
            device->allocated_memory_size[heap_index] -= alloc_size;
            mtx_unlock(&device->overallocation_mutex);
         }
         goto fail;
      }

      if (flags & RADEON_FLAG_GFX12_ALLOW_DCC) {
         if (mem->image) {
            /* Set BO metadata (including DCC tiling flags) for dedicated
             * allocations because compressed writes are enabled and the kernel
             * requires a DCC view for recompression.
             */
            radv_image_bo_set_metadata(device, mem->image, mem->bo);
         } else {
            /* Otherwise, disable compressed writes to prevent recompression
             * when the BO is moved back to VRAM because it's not yet possible
             * to set DCC tiling flags per range for suballocations. The only
             * problem is that we will loose DCC after migration but that
             * should happen rarely.
             */
            struct radeon_bo_metadata md = {0};

            md.u.gfx12.dcc_write_compress_disable = true;

            device->ws->buffer_set_metadata(device->ws, mem->bo, &md);
         }
      }

      mem->heap_index = heap_index;
      mem->alloc_size = alloc_size;

      radv_va_validation_update_page(device, radv_buffer_get_va(mem->bo), alloc_size, true);
   }

   if (!wsi_info) {
      result = device->ws->buffer_make_resident(device->ws, mem->bo, true);
      if (result != VK_SUCCESS)
         goto fail;
   }

   *pMem = radv_device_memory_to_handle(mem);
   radv_rmv_log_heap_create(device, *pMem, is_internal, mem->vk.alloc_flags);

   vk_device_memory_report_emit(&device->vk, result, /* is_alloc */ true, mem->vk.import_handle_type != 0,
                                result == VK_SUCCESS ? mem->bo->obj_id : 0, result == VK_SUCCESS ? mem->bo->size : 0,
                                VK_OBJECT_TYPE_DEVICE_MEMORY, (uintptr_t)mem, mem->heap_index);

   if (import_info)
      close(import_info->fd);

   return VK_SUCCESS;

fail:
   vk_device_memory_report_emit(&device->vk, result, /* is_alloc */ true, mem->vk.import_handle_type != 0,
                                0 /* obj_id */, 0 /* obj_size */, VK_OBJECT_TYPE_DEVICE_MEMORY, (uintptr_t)mem,
                                mem->heap_index);
   radv_free_memory(device, pAllocator, mem);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_AllocateMemory(VkDevice _device, const VkMemoryAllocateInfo *pAllocateInfo,
                    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   return radv_alloc_memory(device, pAllocateInfo, pAllocator, pMem, false);
}

VKAPI_ATTR void VKAPI_CALL
radv_FreeMemory(VkDevice _device, VkDeviceMemory _mem, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_device_memory, mem, _mem);

   if (mem)
      vk_device_memory_report_emit(&device->vk, VK_SUCCESS, /* is_alloc */ false, mem->vk.import_handle_type != 0,
                                   mem->bo->obj_id, mem->bo->size, VK_OBJECT_TYPE_DEVICE_MEMORY, (uintptr_t)mem,
                                   mem->heap_index);

   radv_free_memory(device, pAllocator, mem);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_MapMemory2(VkDevice _device, const VkMemoryMapInfo *pMemoryMapInfo, void **ppData)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_device_memory, mem, pMemoryMapInfo->memory);
   void *fixed_address = NULL;
   bool use_fixed_address = false;

   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) {
      const VkMemoryMapPlacedInfoEXT *placed_info =
         vk_find_struct_const(pMemoryMapInfo->pNext, MEMORY_MAP_PLACED_INFO_EXT);
      if (placed_info) {
         fixed_address = placed_info->pPlacedAddress;
         use_fixed_address = true;
      }
   }

   if (mem->vk.host_ptr)
      *ppData = mem->vk.host_ptr;
   else
      *ppData = device->ws->buffer_map(device->ws, mem->bo, use_fixed_address, fixed_address);

   if (*ppData) {
      vk_rmv_log_cpu_map(&device->vk, radv_buffer_get_va(mem->bo), false);
      *ppData = (uint8_t *)*ppData + pMemoryMapInfo->offset;
      return VK_SUCCESS;
   }

   return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_UnmapMemory2(VkDevice _device, const VkMemoryUnmapInfo *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_device_memory, mem, pMemoryUnmapInfo->memory);

   vk_rmv_log_cpu_map(&device->vk, radv_buffer_get_va(mem->bo), true);
   if (mem->vk.host_ptr == NULL)
      device->ws->buffer_unmap(device->ws, mem->bo, (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT));

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_FlushMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR uint64_t VKAPI_CALL
radv_GetDeviceMemoryOpaqueCaptureAddress(VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   VK_FROM_HANDLE(radv_device_memory, mem, pInfo->memory);
   return radv_buffer_get_va(mem->bo);
}

VKAPI_ATTR void VKAPI_CALL
radv_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize *pCommittedMemoryInBytes)
{
   *pCommittedMemoryInBytes = 0;
}
