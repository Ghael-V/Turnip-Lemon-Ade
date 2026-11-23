/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_image_layout.h"

#include "kk_debug.h"
#include "kk_device.h"
#include "kk_format.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_format.h"

#include "util/format/u_format.h"

static enum mtl_texture_type
vk_image_to_mtl_texture_type(const struct vk_image *image,
                             VkImageUsageFlags usage)
{
   uint32_t array_layers = image->array_layers;
   uint32_t samples = image->samples;
   switch (image->image_type) {
   case VK_IMAGE_TYPE_1D:
   case VK_IMAGE_TYPE_2D:
      /* We require input attachments to be arrays */
      if (array_layers > 1 || (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
         return samples > 1u ? MTL_TEXTURE_TYPE_2D_ARRAY_MULTISAMPLE
                             : MTL_TEXTURE_TYPE_2D_ARRAY;
      return samples > 1u ? MTL_TEXTURE_TYPE_2D_MULTISAMPLE
                          : MTL_TEXTURE_TYPE_2D;
   case VK_IMAGE_TYPE_3D:
      return MTL_TEXTURE_TYPE_3D;
   default:
      UNREACHABLE("Invalid image type");
      return MTL_TEXTURE_TYPE_1D; /* Just return a type we don't actually use */
   }
}

static enum mtl_texture_usage
vk_image_usage_flags_to_mtl_texture_usage(VkImageUsageFlags usage_flags,
                                          VkImageCreateFlags create_flags,
                                          bool supports_atomics)
{
   enum mtl_texture_usage usage = 0u;

   if (usage_flags & VK_IMAGE_USAGE_STORAGE_BIT)
      usage |= MTL_TEXTURE_USAGE_SHADER_WRITE;

   const VkImageUsageFlags shader_read =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
   if (usage_flags & shader_read)
      usage |= MTL_TEXTURE_USAGE_SHADER_READ;

   const VkImageUsageFlags render_attachment =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT;

   if (usage_flags & render_attachment)
      usage |= MTL_TEXTURE_USAGE_RENDER_TARGET;

   if (create_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)
      usage |= MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;

   if (supports_atomics) {
      usage |= MTL_TEXTURE_USAGE_SHADER_READ;
      usage |= MTL_TEXTURE_USAGE_SHADER_WRITE;
      usage |= MTL_TEXTURE_USAGE_SHADER_ATOMIC;
   }

   return usage;
}

bool
kk_image_layout_can_optimize(const struct kk_physical_device *pdev,
                             VkImageUsageFlags usage, VkImageTiling tiling,
                             VkImageCreateFlags flags, enum pipe_format format)
{
   /* Can only optimize if tiling is optimal */
   if (tiling != VK_IMAGE_TILING_OPTIMAL)
      return false;

   /* Cannot optimize if host transfer for a format that would use Apple's
    * lossless compression. Otherwise, CTS tests which populate memory with
    * random data fail due to differences in how invalid optimized data is
    * decompressed by GPU vs CPU. */
   if ((usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT) &&
       !util_format_is_compressed(format))
      return false;

   if (!KK_EXPERIMENTAL(OPTIMIZE_E5B9G9R9) &&
       format == PIPE_FORMAT_R9G9B9E5_FLOAT &&
       (usage &
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)))
      return false;

   /* Attachment feedback usage may produce incorrect results with
    * optimization, causing flakes in CTS
    *
    * E.g. `dEQP-VK.pipeline.monolithic.attachment_feedback_loop_layout.sampler.
    * attachment_feedback_loop_optimal.sampled_image.image_type.2d_unnormalized.
    * format.d16_unorm_depth_read_write_different_areas_dynamic_bad_static`
    */
   if (usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
      return false;

   /* Optimization is disabled for block texel view to ensure we can correctly
    * locate and alias each subresource */
   if (flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT)
      return false;

   return true;
}

static void
kk_image_layout_calc_linear(const struct kk_device *dev,
                            struct kk_image_layout *layout)
{
   size_t bytes_per_texel = util_format_get_blocksize(layout->format.pipe);

   layout->align_B = mtl_minimum_linear_texture_alignment_for_pixel_format(
      dev->mtl_handle, layout->format.mtl);
   layout->linear_stride_B =
      align(bytes_per_texel * layout->width_px, layout->align_B);
   layout->layer_stride_B = layout->linear_stride_B * layout->height_px;
   /* Metal only allows for 2D texture with no mipmapping. */
   layout->size_B = layout->layer_stride_B;
}

static void
kk_image_layout_calc_tiled(const struct kk_device *dev,
                           struct kk_image_layout *layout)
{
   mtl_heap_texture_size_and_align_with_descriptor(
      dev->mtl_handle, layout, &layout->size_B, &layout->align_B);

   /* Stop here if layer stride is not validated for below calculations */
   if (!kk_image_layout_layer_stride_defined(layout))
      return;

   uint32_t layer_count = layout->layers;
   if (layout->type == MTL_TEXTURE_TYPE_3D) {
      /* 3D texture are laid out as 2D array textures */
      layer_count = layout->depth_px;
      layout->layer_stride_B = layout->size_B / layer_count;
   } else if (layout->layers != 1) {
      layout->layers = 1;
      mtl_heap_texture_size_and_align_with_descriptor(
         dev->mtl_handle, layout, &layout->layer_stride_B, NULL);
      layout->layers = layer_count;
   } else {
      /* For one layer, stride is same as image size */
      layout->layer_stride_B = layout->size_B;
   }

   /* Layer stride times number of layers should equal total size. */
   assert(layout->layer_stride_B * layer_count == layout->size_B);
}

static void
kk_image_layout_calc_offsets(const struct kk_device *dev,
                             struct kk_image_layout *layout)
{
   /* Skip if level offsets are not validated for below calculations */
   if (!kk_image_layout_level_offsets_defined(layout))
      return;

   if (layout->linear) {
      /* We only support one level for linear textures */
      layout->level_offsets_B[0] = 0;
      layout->level_offsets_B[1] = layout->layer_stride_B;
      return;
   }

   struct kk_image_layout calc_layout = *layout;
   calc_layout.layers = 1;
   calc_layout.levels = 1;

   /* 3D texture are laid out as 2D array textures */
   if (calc_layout.type == MTL_TEXTURE_TYPE_3D) {
      calc_layout.type = MTL_TEXTURE_TYPE_2D;
      calc_layout.depth_px = 1u;
   }

   /* Default sparse tile size follows regular tile size */
   uint32_t tile_size_B = mtl_sparse_tile_size_in_bytes(dev->mtl_handle);
   struct mtl_size tile_size_el =
      mtl_sparse_tile_size(dev->mtl_handle, &calc_layout);
   struct mtl_size tile_count =
      mtl_sparse_tile_count(dev->mtl_handle, &calc_layout, tile_size_el);

   layout->level_offsets_B[0] = 0;

   /* We also add the end offset so we can easily recover the size of a level */
   assert(layout->levels < ARRAY_SIZE(layout->level_offsets_B));

   /* Calculate which level the small power-of-two miptree begins at. The
    * power-of-two miptree is used when either the width or the height is
    * smaller than a single large tile. Same calculation as HoneyKrisp.
    *
    * MTLTexture.firstMipmapInTail is only defined for sparse textures.
    * macOS 26 always returns 0 if the texture is not sparse so it cannot be
    * used for non-sparse textures. macOS 27 gives us the value we require for
    * the computations.
    */
   enum pipe_format p_format = layout->format.pipe;
   uint32_t w_el = util_format_get_nblocksx(p_format, layout->width_px);
   uint32_t h_el = util_format_get_nblocksy(p_format, layout->height_px);
   uint32_t bw_px = util_format_get_blockwidth(p_format);
   uint32_t bh_px = util_format_get_blockheight(p_format);
   uint32_t pot_level = 0;
   uint32_t pot_w_px = bw_px * w_el;
   uint32_t pot_h_px = bh_px * h_el;
   uint32_t max_blocks_w = tile_size_el.x / bw_px;
   uint32_t max_blocks_h = tile_size_el.y / bh_px;
   do {
      uint32_t pot_w_el = util_format_get_nblocksx(p_format, pot_w_px);
      uint32_t pot_h_el = util_format_get_nblocksy(p_format, pot_h_px);
      if (pot_w_el < max_blocks_w || pot_h_el < max_blocks_h)
         break;
      pot_w_px = u_minify(pot_w_px, 1);
      pot_h_px = u_minify(pot_h_px, 1);
      pot_level++;
   } while (1);

   for (uint8_t level = 0; level < layout->levels; ++level) {
      calc_layout.width_px = u_minify(layout->width_px, level);
      calc_layout.height_px = u_minify(layout->height_px, level);

      uint64_t level_size;
      mtl_heap_texture_size_and_align_with_descriptor(
         dev->mtl_handle, &calc_layout, &level_size, NULL);

      /* Based on HoneyKrisp layout calculations. There may be a padding corner
       * tile, which appears to be excluded by Metal calculations when querying
       * for a single mip level. Add its size on if needed. This only applies
       * to mip levels before the tail. */
      if (level < pot_level) {
         uint32_t mip_tiles = (tile_count.x * tile_count.y) >> (level * 2);
         bool pad_left = tile_count.x & BITFIELD_MASK(level);
         bool pad_bottom = tile_count.y & BITFIELD_MASK(level);
         bool pad_corner = pad_left && pad_bottom;
         if (mip_tiles != 0 && pad_corner)
            level_size += tile_size_B;
      }

      layout->level_offsets_B[level + 1] =
         layout->level_offsets_B[level] + level_size;
   }

   /* End of last mip level should never exceed layer stride, but may be less
    * due to extra padding */
   assert(layout->level_offsets_B[layout->levels] <= layout->layer_stride_B);
}

void
kk_image_layout_init(const struct kk_device *dev, const struct vk_image *image,
                     enum pipe_format format, const uint8_t width_scale,
                     const uint8_t height_scale, struct kk_image_layout *layout)
{
   const struct kk_physical_device *pdev = kk_device_physical(dev);
   const struct kk_va_format *supported_format = kk_get_va_format(format);
   VkImageUsageFlags usage = vk_image_usage(image, image->aspects);
   layout->type = vk_image_to_mtl_texture_type(image, usage);
   layout->width_px = image->extent.width / width_scale;
   layout->height_px = image->extent.height / height_scale;
   layout->depth_px = image->extent.depth;
   layout->layers = image->array_layers;
   layout->levels = image->mip_levels;
   layout->linear = image->tiling != VK_IMAGE_TILING_OPTIMAL;
   layout->optimized_layout = kk_image_layout_can_optimize(
      pdev, usage, image->tiling, image->create_flags, format);
   layout->usage = vk_image_usage_flags_to_mtl_texture_usage(
      usage, image->create_flags, supported_format->atomic);
   layout->create_flags = image->create_flags;
   layout->format.pipe = format;
   layout->format.mtl = supported_format->mtl_pixel_format;
   layout->swizzle.red = supported_format->unswizzle.red;
   layout->swizzle.green = supported_format->unswizzle.green;
   layout->swizzle.blue = supported_format->unswizzle.blue;
   layout->swizzle.alpha = supported_format->unswizzle.alpha;
   layout->sample_count_sa = image->samples;

   /*
    * Metal requires adding MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW if we are going
    * to reinterpret the format with a different format. This seems to be the
    * only format with this issue.
    */
   if (format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
      layout->usage |= MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;
   }

   if (layout->linear) {
      kk_image_layout_calc_linear(dev, layout);
   } else {
      kk_image_layout_calc_tiled(dev, layout);
   }

   kk_image_layout_calc_offsets(dev, layout);
}
