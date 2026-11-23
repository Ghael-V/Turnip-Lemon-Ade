/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_XFB_H
#define PVR_XFB_H

#include <stdbool.h>
#include <stdint.h>

#include "pvr_limits.h"
#include "pvr_types.h"

struct pvr_device;
struct pvr_physical_device;
struct pvr_cmd_buffer;
struct pvr_query_pool;
struct pvr_suballoc_bo;

struct pvr_pds_xfb_buffer_info {
   pvr_dev_addr_t buffer_address;
   pvr_dev_addr_t buffer_limit;
};

struct pvr_pds_xfb_varying_info {
   uint32_t varying_index;
   uint32_t varying_stride;
   uint32_t dma_count;
   uint32_t output_offset;
};

struct pvr_xfb_program_input {
   uint16_t primitive_type;
   uint32_t varying_count;

   struct {
      struct pvr_pds_xfb_varying_info *varying_info_arr;
      struct pvr_pds_xfb_buffer_info *buffer_info;
   } data;
};

struct pvr_pds_xfb_init_program_input {
   bool load_from_counter_buffers;

   struct pvr_device *device;

   uint16_t primitive_type;

   uint32_t counter_buffer_count;

   struct {
      pvr_dev_addr_t *counter_buffer_offsets_addr;

      struct pvr_pds_xfb_varying_info *varying_info_arr;
      struct pvr_pds_xfb_buffer_info *buffer_info;

      pvr_dev_addr_t *counter_buffers_addr;

      struct pvr_suballoc_bo
         *ptemp_memory_buffer_addr_array[PVR_XFB_MAX_BUFFERS];
   } data;
};

struct pvr_pds_xfb_end_program_input {
   bool has_store_counter_buffers;
   uint32_t buffer_strides[PVR_XFB_MAX_BUFFERS];
   uint32_t counter_buffer_count;

   struct {
      pvr_dev_addr_t *counter_buffers_addr;
      pvr_dev_addr_t *counter_buffer_offsets_addr;
   } data;
};

struct pvr_pds_xfb_draw_program_input {
   uint32_t first_instance;
   uint32_t instance_count;
   uint32_t vertex_stride;

   uint32_t counter_offset;
   struct {
      pvr_dev_addr_t counter_buffer_addr;
      pvr_dev_addr_t output_count_buffer_addr;
   } data;
};

struct pvr_pds_xfb_query_program_input {
   bool is_active;

   pvr_dev_addr_t result_addr;
   pvr_dev_addr_t availability_addr;

   pvr_dev_addr_t count_offset_addr;
};

/* pvr_xfb_get_counter_dwords - return the counter size in dword. */
static inline uint32_t
pvr_xfb_get_counter_dwords(const struct pvr_device_info *const dev_info)
{
   const uint32_t bits =
      PVR_GET_FEATURE_VALUE(dev_info, transform_feedback_query_bits, 32U);
   return bits == 64U ? 2 : 1;
}

/* pvr_xfb_get_counter_bytes - return the counter size in bytes. */
static inline uint32_t
pvr_xfb_get_counter_bytes(const struct pvr_device_info *const dev_info)
{
   const uint32_t bits =
      PVR_GET_FEATURE_VALUE(dev_info, transform_feedback_query_bits, 32U);
   return bits == 64U ? 8 : 4;
}

#define PVR_XFB_QUERY_RESULT_ALIGN 16

#ifdef PVR_PER_ARCH

void PVR_PER_ARCH(xfb_query_pool_begin)(struct pvr_cmd_buffer *cmd_buffer,
                                        struct pvr_query_pool *pool,
                                        uint32_t query_idx);

void PVR_PER_ARCH(xfb_query_pool_end)(struct pvr_cmd_buffer *cmd_buffer,
                                      struct pvr_query_pool *pool,
                                      uint32_t query_idx);

#endif /* PVR_PER_ARCH */
#endif /* PVR_XFB_H */
