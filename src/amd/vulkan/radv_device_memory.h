/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_DEVICE_MEMORY_H
#define RADV_DEVICE_MEMORY_H

#include "vk_device_memory.h"

#include "radv_android.h"

struct radv_device;

struct radv_device_memory {
   struct vk_device_memory vk;
   struct radeon_winsys_bo *bo;
   /* for dedicated allocations */
   struct radv_image *image;
   struct radv_buffer *buffer;
   uint32_t heap_index;
   uint64_t alloc_size;
   void *map;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(radv_device_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)

void radv_free_memory(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                      struct radv_device_memory *mem);

VkResult radv_alloc_memory(struct radv_device *device, const VkMemoryAllocateInfo *pAllocateInfo,
                           const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem, bool is_internal);

#endif /* RADV_DEVICE_MEMORY_H */
