/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_FORMAT_H
#define NVK_FORMAT_H 1

#include "nvk_private.h"
#include "vk_format.h"

#include "util/format/u_formats.h"
#include "nv_device_info.h"

struct nvk_physical_device;

#define NVK_VA_BIT_WIDTH_NONE 0

struct nvk_va_format {
   uint8_t bit_widths;
   uint8_t bit_widths_high;
   uint8_t swap_rb:1;
   uint8_t type:7;
};

bool
nvk_format_supports_atomics(const struct nvk_physical_device *pdev,
                            enum pipe_format p_format);

bool
nvk_format_supports_storage(const struct nvk_physical_device *pdev,
                            enum pipe_format p_format);

const struct nvk_va_format *
nvk_get_va_format(const struct nvk_physical_device *pdev, VkFormat format);

static inline enum pipe_format
nvk_format_to_pipe_format(VkFormat vkformat)
{
   return vk_format_to_pipe_format(vkformat);
}

#endif
