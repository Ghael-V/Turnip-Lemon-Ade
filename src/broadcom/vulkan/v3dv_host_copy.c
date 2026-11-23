/*
 * Copyright © 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "v3dv_device.h"
#include "v3dv_entrypoints.h"
#include "v3dv_image.h"

#include "util/box.h"
#include "vk_format.h"
#include "vk_log.h"

/* A rectangle in blocks (== texels for uncompressed formats). */
struct blk_rect {
   int32_t x, y;
   uint32_t w, h;
};

static struct blk_rect
texels_to_blocks(const struct v3dv_image *image, uint8_t plane,
                 VkOffset3D off, VkExtent3D ext)
{
   const VkFormat f = image->planes[plane].vk_format;
   const uint32_t bw = vk_format_get_blockwidth(f);
   const uint32_t bh = vk_format_get_blockheight(f);
   return (struct blk_rect) {
      .x = off.x / bw,
      .y = off.y / bh,
      .w = DIV_ROUND_UP(ext.width, bw),
      .h = DIV_ROUND_UP(ext.height, bh),
   };
}

/* Returns the CPU address of the start of the memory object bound to a plane.
 * We never unmap: the driver can map BOs internally and the app may have
 * mapped the same memory.
 */
static VkResult
map_plane(struct v3dv_device *device, const struct v3dv_image *image,
          uint8_t plane, uint8_t **base)
{
   struct v3dv_device_memory *mem = image->planes[plane].mem;
   if (!mem || !mem->bo)
      return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);

   if (!mem->bo->map && !v3dv_bo_map(device, mem->bo, mem->bo->size))
      return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);

   *base = mem->bo->map;
   return VK_SUCCESS;
}

/* "Slices" are depth slices for 3D images and array layers otherwise. */
static void
slice_range(const struct v3dv_image *image, const VkImageSubresourceLayers *sub,
            VkOffset3D offset, VkExtent3D extent,
            uint32_t *first, uint32_t *count)
{
   if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
      *first = offset.z;
      *count = extent.depth;
   } else {
      *first = sub->baseArrayLayer;
      *count = vk_image_subresource_layer_count(&image->vk, sub);
   }
}

/* Copies a block rectangle of `slice_count` slices between the image and a
 * linear host buffer. `host` is the top-left of the rectangle in the first
 * slice.
 */
static VkResult
copy_blocks(struct v3dv_device *device, struct v3dv_image *image,
            uint8_t plane, uint32_t level, struct blk_rect r,
            uint32_t first_slice, uint32_t slice_count,
            void *host, uint32_t host_row_stride, size_t host_slice_stride,
            bool to_image, VkImageAspectFlags aspect)
{
   const uint32_t cpp = image->planes[plane].cpp;   /* bytes per block */
   const struct v3d_resource_slice *slice =
      &image->planes[plane].slices[level];

   uint8_t *base;
   VkResult result = map_plane(device, image, plane, &base);
   if (result != VK_SUCCESS)
      return result;

   struct pipe_box box = {
      .x = r.x, .y = r.y, .width = r.w, .height = r.h,
   };

   /* aspect != 0: single-aspect copy of a D24S8 image. The host texel is
    * 4 bytes (X8_D24) for depth and 1 byte (S8) for stencil, while the
    * image texel is always 4 bytes, so we stage through a linear temp. */
   const bool partial = aspect != 0;
   const uint32_t host_cpp = aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1 : 4;
   const uint32_t tmp_stride = r.w * cpp;
   uint8_t *tmp = NULL;

   if (partial) {
      tmp = vk_alloc(&device->vk.alloc, (size_t)tmp_stride * r.h, 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (!tmp)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t i = 0; i < slice_count; i++) {
      /* v3dv_layer_offset() already includes planes[plane].mem_offset */
      uint8_t *gpu = base + v3dv_layer_offset(image, level,
                                              first_slice + i, plane);
      uint8_t *cpu = (uint8_t *)host + i * host_slice_stride;

      /* Linear side of the tiled/raster transfer. */
      uint8_t *lin = partial ? tmp : cpu;
      const uint32_t lin_stride = partial ? tmp_stride : host_row_stride;

      /* Image -> linear. For partial host->image copies this is still
       * needed so the other aspect is preserved. */
      if (!to_image || partial) {
         if (slice->tiling == V3D_TILING_RASTER) {
            for (uint32_t row = 0; row < r.h; row++) {
               memcpy(lin + (size_t)row * lin_stride,
                      gpu + (size_t)(r.y + row) * slice->stride +
                      (size_t)r.x * cpp,
                      (size_t)r.w * cpp);
            }
         } else {
            v3d_load_tiled_image(lin, lin_stride, gpu, slice->stride,
                                 slice->tiling, cpp, slice->padded_height,
                                 &box);
         }
      }

      /* Convert between the two texel layouts involved:
       *
       *  - tmp (staging, copied raw from the GPU) uses V3D's hardware
       *    layout: stencil in bits 0-7, depth in bits 8-31.
       *  - host (the app's memory) uses Vulkan's layout: depth in bits
       *    0-23 and, for D24_UNORM_S8_UINT, stencil in bits 24-31.
       *    Depth-aspect host texels are 4 bytes (X8_D24, top byte
       *    unused) and stencil-aspect host texels are 1 byte (S8).
       *
       * Only the selected aspect is touched, so for host->image the other
       * aspect (or the padding byte of X8_D24) is preserved from the
       * texels we read back into tmp above.
       */
      if (partial) {
         for (uint32_t row = 0; row < r.h; row++) {
            uint8_t *t = tmp + (size_t)row * tmp_stride;
            uint8_t *h = cpu + (size_t)row * host_row_stride;

            for (uint32_t x = 0; x < r.w; x++) {
               uint8_t *tx = t + (size_t)x * 4;
               uint8_t *hx = h + (size_t)x * host_cpp;
               uint32_t texel;
               memcpy(&texel, tx, 4);

               if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
                  if (to_image) {
                     uint32_t d;
                     memcpy(&d, hx, 4);
                     texel = (texel & 0x000000ffu) | ((d & 0x00ffffffu) << 8);
                     memcpy(tx, &texel, 4);
                  } else {
                     uint32_t d = texel >> 8;
                     memcpy(hx, &d, 4);
                  }
               } else {
                  assert(aspect == VK_IMAGE_ASPECT_STENCIL_BIT);
                  if (to_image) {
                     texel = (texel & 0xffffff00u) | *hx;
                     memcpy(tx, &texel, 4);
                  } else {
                     *hx = (uint8_t)(texel & 0xffu);
                  }
               }
            }
         }
      }

      /* Linear -> image. */
      if (to_image) {
         if (slice->tiling == V3D_TILING_RASTER) {
            for (uint32_t row = 0; row < r.h; row++) {
               memcpy(gpu + (size_t)(r.y + row) * slice->stride +
                      (size_t)r.x * cpp,
                      lin + (size_t)row * lin_stride,
                      (size_t)r.w * cpp);
            }
         } else {
            v3d_store_tiled_image(gpu, slice->stride, lin, lin_stride,
                                  slice->tiling, cpp, slice->padded_height,
                                  &box);
         }
      }
   }

   vk_free(&device->vk.alloc, tmp);
   return VK_SUCCESS;
}

/* Size in bytes of one memcpy'd subresource, as reported by
 * vkGetImageSubresourceLayout2.
 */
static size_t
subresource_memcpy_size(struct v3dv_device *device, struct v3dv_image *image,
                        VkImageAspectFlags aspect, uint32_t level)
{
   const VkImageSubresource2KHR sub2 = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR,
      .imageSubresource = {
         .aspectMask = aspect, .mipLevel = level, .arrayLayer = 0,
      },
   };
   VkSubresourceLayout2KHR layout2 = {
      .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR,
   };
   v3dv_GetImageSubresourceLayout2KHR(v3dv_device_to_handle(device),
                                      v3dv_image_to_handle(image),
                                      &sub2, &layout2);
   return layout2.subresourceLayout.size;
}

/* VK_HOST_IMAGE_COPY_MEMCPY_BIT: copy the raw bytes of each subresource.
 * The spec doesn't say how several layers are arranged in host memory, so
 * they are packed back to back, VkSubresourceHostMemcpySize bytes each.
 */
static VkResult
memcpy_layers(struct v3dv_device *device, struct v3dv_image *image,
              const VkImageSubresourceLayers *sub, void *host, bool to_image)
{
   const uint8_t plane = v3dv_plane_from_aspect(sub->aspectMask);

   uint8_t *base;
   VkResult result = map_plane(device, image, plane, &base);
   if (result != VK_SUCCESS)
      return result;

   const size_t size =
      subresource_memcpy_size(device, image, sub->aspectMask, sub->mipLevel);
   const uint32_t layers = vk_image_subresource_layer_count(&image->vk, sub);

   /* 3D images are copied as a single subresource. */
   assert(image->vk.image_type != VK_IMAGE_TYPE_3D ||
          (sub->baseArrayLayer == 0 && layers == 1));

   for (uint32_t i = 0; i < layers; i++) {
      uint8_t *gpu = base + v3dv_layer_offset(image, sub->mipLevel,
                                              sub->baseArrayLayer + i, plane);
      uint8_t *cpu = (uint8_t *)host + i * size;
      if (to_image)
         memcpy(gpu, cpu, size);
      else
         memcpy(cpu, gpu, size);
   }

   return VK_SUCCESS;
}

static VkResult
copy_memory_region(struct v3dv_device *device, struct v3dv_image *image,
                   const VkImageSubresourceLayers *sub,
                   VkOffset3D offset, VkExtent3D extent,
                   const struct vk_image_buffer_layout *mem_layout,
                   void *host, VkHostImageCopyFlags flags, bool to_image)
{
   if (flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT)
      return memcpy_layers(device, image, sub, host, to_image);

   const uint8_t plane = v3dv_plane_from_aspect(sub->aspectMask);

   const bool partial =
      (image->vk.format == VK_FORMAT_D24_UNORM_S8_UINT &&
       (sub->aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT ||
        sub->aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)) ||
      image->vk.format == VK_FORMAT_X8_D24_UNORM_PACK32;

   assert(partial ||
          mem_layout->element_size_B == image->planes[plane].cpp);

   uint32_t first, count;
   slice_range(image, sub, offset, extent, &first, &count);

   /* Host memory makes no distinction between array layers and depth
    * slices, image_stride_B applies to both. */
   return copy_blocks(device, image, plane, sub->mipLevel,
                      texels_to_blocks(image, plane, offset, extent),
                      first, count,
                      host, mem_layout->row_stride_B,
                      mem_layout->image_stride_B, to_image,
                      partial ? sub->aspectMask : 0);
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyMemoryToImage(VkDevice _device, const VkCopyMemoryToImageInfo *pInfo)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, image, pInfo->dstImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkMemoryToImageCopy *r = &pInfo->pRegions[i];
      const struct vk_image_buffer_layout layout =
         vk_memory_to_image_copy_layout(&image->vk, r);
      VkResult result =
         copy_memory_region(device, image, &r->imageSubresource,
                            r->imageOffset, r->imageExtent, &layout,
                            (void *)r->pHostPointer, pInfo->flags, true);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyImageToMemory(VkDevice _device, const VkCopyImageToMemoryInfo *pInfo)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, image, pInfo->srcImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageToMemoryCopy *r = &pInfo->pRegions[i];
      const struct vk_image_buffer_layout layout =
         vk_image_to_memory_copy_layout(&image->vk, r);
      VkResult result =
         copy_memory_region(device, image, &r->imageSubresource,
                            r->imageOffset, r->imageExtent, &layout,
                            r->pHostPointer, pInfo->flags, false);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

/* Image to image goes through a ~64KiB linear temporary, band by band:
 * bounded memory, and correct even when src and dst are the same image.
 */
VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyImageToImage(VkDevice _device, const VkCopyImageToImageInfo *pInfo)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, src, pInfo->srcImage);
   V3DV_FROM_HANDLE(v3dv_image, dst, pInfo->dstImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageCopy2 *r = &pInfo->pRegions[i];

      const uint8_t src_plane =
         v3dv_plane_from_aspect(r->srcSubresource.aspectMask);
      const uint8_t dst_plane =
         v3dv_plane_from_aspect(r->dstSubresource.aspectMask);

      const VkImageAspectFlags src_aspect = r->srcSubresource.aspectMask;
      const VkImageAspectFlags dst_aspect = r->dstSubresource.aspectMask;

      /* Single-aspect copies of D24S8 go through copy_blocks' partial path. */
      const bool src_partial =
         (src->vk.format == VK_FORMAT_D24_UNORM_S8_UINT &&
          (src_aspect == VK_IMAGE_ASPECT_DEPTH_BIT ||
           src_aspect == VK_IMAGE_ASPECT_STENCIL_BIT)) ||
         src->vk.format == VK_FORMAT_X8_D24_UNORM_PACK32;
      const bool dst_partial =
         (dst->vk.format == VK_FORMAT_D24_UNORM_S8_UINT &&
          (dst_aspect == VK_IMAGE_ASPECT_DEPTH_BIT ||
           dst_aspect == VK_IMAGE_ASPECT_STENCIL_BIT)) ||
         dst->vk.format == VK_FORMAT_X8_D24_UNORM_PACK32;

      /* Bytes per texel in the tmp buffer: the single-aspect host size when
       * partial (4 for X8_D24, 1 for S8), otherwise the plane's block size. */
      const uint32_t src_cpp =
         src_partial ? (src_aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1 : 4)
                     : src->planes[src_plane].cpp;
      const uint32_t dst_cpp =
         dst_partial ? (dst_aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1 : 4)
                     : dst->planes[dst_plane].cpp;
      assert(src_cpp == dst_cpp);
      const uint32_t cpp = src_cpp;

      uint32_t src_first, src_count, dst_first, dst_count;
      slice_range(src, &r->srcSubresource, r->srcOffset, r->extent,
                  &src_first, &src_count);
      slice_range(dst, &r->dstSubresource, r->dstOffset, r->extent,
                  &dst_first, &dst_count);
      const uint32_t count = MIN2(src_count, dst_count);

      const bool d24s8_single_aspect =
         src->vk.format == VK_FORMAT_D24_UNORM_S8_UINT &&
         (src_aspect == VK_IMAGE_ASPECT_DEPTH_BIT ||
          src_aspect == VK_IMAGE_ASPECT_STENCIL_BIT);

      if ((pInfo->flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT) &&
          !d24s8_single_aspect) {
         /* Raw copy. Sizes must match. */
         uint8_t *src_base, *dst_base;
         VkResult res = map_plane(device, src, src_plane, &src_base);
         if (res == VK_SUCCESS)
            res = map_plane(device, dst, dst_plane, &dst_base);
         if (res != VK_SUCCESS)
            return res;

         const size_t size = subresource_memcpy_size(
            device, src, r->srcSubresource.aspectMask,
            r->srcSubresource.mipLevel);
         assert(size == subresource_memcpy_size(
                   device, dst, r->dstSubresource.aspectMask,
                   r->dstSubresource.mipLevel));

         const bool is_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
         const uint32_t n = is_3d ? 1 : count;
         /* VUID-VkCopyImageToImageInfo-{src,dst}Offset-09114 and
          * VUID-VkCopyImageToImageInfo-{src,dst}Image-09115: with
          * VK_HOST_IMAGE_COPY_MEMCPY_BIT the offsets are 0 and the extent covers
          * the whole subresource. For 3D images VkSubresourceLayout::size spans
          * all depth slices, so a single memcpy from slice 0 is exact and
          * offset.z / extent.depth need no special handling.
          */
         for (uint32_t s = 0; s < n; s++) {
            memcpy(dst_base + v3dv_layer_offset(dst, r->dstSubresource.mipLevel,
                                                is_3d ? 0 : dst_first + s,
                                                dst_plane),
                   src_base + v3dv_layer_offset(src, r->srcSubresource.mipLevel,
                                                is_3d ? 0 : src_first + s,
                                                src_plane),
                   size);
         }
         continue;
      }

      /* The extent is in units of the source image, also when copying
       * between compressed and uncompressed formats, so both rectangles
       * cover the same number of blocks. Only the destination origin is
       * converted with the destination's block size.
       */
      const struct blk_rect sr =
         texels_to_blocks(src, src_plane, r->srcOffset, r->extent);
      struct blk_rect dr =
         texels_to_blocks(dst, dst_plane, r->dstOffset, r->extent);
      assert(dr.w == sr.w && dr.h == sr.h);

      const size_t row_B = (size_t)sr.w * cpp;
      const uint32_t band_h = MAX2(1u, (uint32_t)(65536 / row_B));

      void *tmp = vk_alloc(&device->vk.alloc, row_B * MIN2(band_h, sr.h), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (!tmp)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

      VkResult result = VK_SUCCESS;
      for (uint32_t s = 0; s < count && result == VK_SUCCESS; s++) {
         for (uint32_t y = 0; y < sr.h; y += band_h) {
            const uint32_t h = MIN2(band_h, sr.h - y);

            struct blk_rect sb = { sr.x, sr.y + y, sr.w, h };
            struct blk_rect db = { dr.x, dr.y + y, dr.w, h };

            result = copy_blocks(device, src, src_plane,
                                 r->srcSubresource.mipLevel, sb,
                                 src_first + s, 1, tmp, row_B, 0, false,
                                 src_partial ? src_aspect : 0);
            if (result == VK_SUCCESS) {
               result = copy_blocks(device, dst, dst_plane,
                                    r->dstSubresource.mipLevel, db,
                                    dst_first + s, 1, tmp, row_B, 0, true,
                                    dst_partial ? dst_aspect : 0);
            }
            if (result != VK_SUCCESS)
               break;
         }
      }

      vk_free(&device->vk.alloc, tmp);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_TransitionImageLayout(VkDevice device, uint32_t transitionCount,
                           const VkHostImageLayoutTransitionInfo *pTransitions)
{
   /* V3D has no layout-dependent metadata: this is a no-op. */
   return VK_SUCCESS;
}
