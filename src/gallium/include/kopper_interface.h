/*
 * Copyright 2020 Red Hat, Inc.
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

/*
 * In principle this could all go in dri_interface.h, but:
 * - I want type safety in here, but I don't want to require vulkan.h from
 *   dri_interface.h
 * - I don't especially want this to be an interface outside of Mesa itself
 * - Ideally dri_interface.h wouldn't even be a thing anymore
 *
 * So instead let's just keep this as a Mesa-internal detail.
 */

#ifndef KOPPER_INTERFACE_H
#define KOPPER_INTERFACE_H

#include "mesa_interface.h"
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct dri_drawable;

typedef struct __DRIkopperLoaderExtensionRec    __DRIkopperLoaderExtension;
typedef struct __DRIkopperDrawableInfoRec    __DRIkopperDrawableInfo;

struct __DRIkopperDrawableInfoRec {
   bool multiplanes_available;
   int is_pixmap;
};

/**
 * Kopper loader extension.
 */

/**
 * struct for storage the union of all platform depdendent
 * Vk*SurfaceCreateInfo* type, all platform Vk*SurfaceCreateInfo* contains
 * uint32_t flags and at most two extra pointer besides bos header.
 * For example:
 * VkWin32SurfaceCreateInfoKHR contains flags, hinstance and hwnd besides bos header
 */

struct kopper_vk_surface_create_storage {
   /* First two fields are copied from VkBaseOutStructure for easily access shared properties */
   VkStructureType sType;
   struct VkBaseOutStructure *pNext;
   intptr_t padding[3];
};

struct kopper_loader_info {
   struct kopper_vk_surface_create_storage bos;
   int has_alpha;
   int initial_swap_interval;
   bool present_opaque;
   uint32_t compression;
};

struct __DRIkopperLoaderExtensionRec {
    /* Asks the loader to fill in VkWhateverSurfaceCreateInfo etc. */
    void (*SetSurfaceCreateInfo)(void *draw, struct kopper_loader_info *out);
    /* Asks the loader to fill in the drawable's width and height */
    void (*GetDrawableInfo)(struct dri_drawable *draw, int *w, int *h,
                            void *closure);
    /* Return a loader capability value. If the loader doesn't know the enum,
     * it will return 0.
     */
    unsigned (*getCapability)(void *loaderPrivate, enum dri_loader_cap cap);
    /* Optional. Returns a serial that changes when the drawable size may have
     * changed, or 0 if the loader cannot tell and the size must always be queried.
     */
    uint32_t (*GetResizeSerial)(void *loaderPrivate);
};
#endif /* KOPPER_INTERFACE_H */
