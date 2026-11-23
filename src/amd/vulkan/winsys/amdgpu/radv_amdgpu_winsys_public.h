/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based on amdgpu winsys.
 * Copyright © 2011 Marek Olšák <maraeo@gmail.com>
 * Copyright © 2015 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_AMDGPU_WINSYS_PUBLIC_H
#define RADV_AMDGPU_WINSYS_PUBLIC_H

#include "ac_gpu_info.h"
#include "ac_linux_drm.h"

#include "vk_sync.h"

enum radv_amdgpu_cs_domain_mode {
   RADV_AMDGPU_CS_DOMAIN_AUTO,
   RADV_AMDGPU_CS_DOMAIN_VRAM,
   RADV_AMDGPU_CS_DOMAIN_GTT,
};

struct radv_amdgpu_winsys_options {
   bool is_virtio;

   /* Debugging */
   bool chain_ib;
   bool debug_all_bos;
   bool debug_log_bos;
   bool debug_vm;
   bool dump_ibs;
   bool dump_bo_history;

   /* Memory placement */
   bool gtt_spill;
   bool always_valid_bos;
   enum radv_amdgpu_cs_domain_mode cs_domain_mode;
};

VkResult radv_amdgpu_winsys_create(int fd, const struct radeon_info *info,
                                   const struct radv_amdgpu_winsys_options *options, struct radeon_winsys **winsys);

struct radeon_winsys_info {
   struct radeon_info base;
   struct vk_sync_type syncobj_sync_type;
   uint32_t global_priority_mask;
};

VkResult radv_amdgpu_winsys_query_info(int fd, bool is_virtio, bool cache_compat, struct radeon_winsys_info *info);

struct radeon_winsys_heap_info {
   uint64_t allocated_vram;
   uint64_t vram_usage;
   uint64_t allocated_vram_vis;
   uint64_t vram_vis_usage;
   uint64_t allocated_gtt;
   uint64_t gtt_usage;
};

int radv_amdgpu_winsys_query_heap_info(ac_drm_device *dev, struct radeon_winsys_heap_info *heap_info);

#endif /* RADV_AMDGPU_WINSYS_PUBLIC_H */
