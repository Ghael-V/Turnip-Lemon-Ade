/*
 * Copyright © 2021 Intel Corporation
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

#ifndef VK_DEBUG_UTILS_H
#define VK_DEBUG_UTILS_H

#include "vk_device.h"
#include "vk_instance.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vk_queue;

struct vk_debug_utils_messenger {
   struct vk_object_base base;
   VkAllocationCallbacks alloc;

   struct list_head link;

   VkDebugUtilsMessageSeverityFlagsEXT severity;
   VkDebugUtilsMessageTypeFlagsEXT type;
   PFN_vkDebugUtilsMessengerCallbackEXT callback;
   void *data;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(vk_debug_utils_messenger, base,
                               VkDebugUtilsMessengerEXT,
                               VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT)

void
vk_debug_message(struct vk_instance *instance,
                 VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                 VkDebugUtilsMessageTypeFlagsEXT types,
                 const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData);

void
vk_debug_message_instance(struct vk_instance *instance,
                          VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                          VkDebugUtilsMessageTypeFlagsEXT types,
                          const char *pMessageIdName,
                          int32_t messageIdNumber,
                          const char *pMessage);

void
vk_address_binding_report(struct vk_instance *instance,
                          struct vk_object_base *object,
                          uint64_t base_address,
                          uint64_t size,
                          VkDeviceAddressBindingTypeEXT type);

static inline void
vk_device_memory_report_emit(struct vk_device* device,
                             VkResult result,
                             bool is_alloc,
                             bool is_import,
                             uint64_t mem_obj_id,
                             VkDeviceSize size,
                             VkObjectType obj_type,
                             uint64_t obj_handle,
                             uint32_t heap_index)
{
   if (likely(!device->memory_reports))
      return;

   VkDeviceMemoryReportEventTypeEXT type;
   if (result != VK_SUCCESS) {
      type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT;
   } else if (is_alloc) {
      type = is_import ? VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT
                       : VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT;
   } else {
      type = is_import ? VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT
                       : VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT;
   }

   const VkDeviceMemoryReportCallbackDataEXT report = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT,
      .type = type,
      .memoryObjectId = mem_obj_id,
      .size = size,
      .objectType = obj_type,
      .objectHandle = obj_handle,
      .heapIndex = heap_index,
   };

   for (uint32_t i = 0; i < device->memory_report_count; i++)
      device->memory_reports[i].callback(&report, device->memory_reports[i].data);
}

void
vk_queue_begin_debug_utils_label(struct vk_queue *queue,
                                 const VkDebugUtilsLabelEXT *pLabelInfo);

void
vk_queue_end_debug_utils_label(struct vk_queue *queue);


struct u_printf_ctx;

VkResult
vk_check_printf_status(struct vk_device *dev, struct u_printf_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* VK_DEBUG_UTILS_H */
