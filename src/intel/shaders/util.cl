/* Copyright © 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "libintel_shaders.h"

/* Memcpy data using multiple lanes. */
void genX(libanv_memcpy)(global void *dst_base,
                         global void *src_base,
                         uint num_dwords,
                         uint dword_offset)
{
   global void *dst = dst_base + 4 * dword_offset;
   global void *src = src_base + 4 * dword_offset;

   if (dword_offset + 4 <= num_dwords) {
      *(global uint4 *)(dst) = *(global uint4 *)(src);
   } else if (dword_offset + 3 <= num_dwords) {
      *(global uint3 *)(dst) = *(global uint3 *)(src);
   } else if (dword_offset + 2 <= num_dwords) {
      *(global uint2 *)(dst) = *(global uint2 *)(src);
   } else if (dword_offset + 1 <= num_dwords) {
      *(global uint *)(dst) = *(global uint *)(src);
   }
}

/* Copy size from src_ptr to dst_ptr for using a single lane with size
 * multiple of 4.
 */
void genX(copy_data)(global void *dst_ptr,
                     global void *src_ptr,
                     uint32_t size)
{
   for (uint32_t offset = 0; offset < size; offset += 16) {
      if (offset + 16 <= size) {
         *(global uint4 *)(dst_ptr + offset) = *(global uint4 *)(src_ptr + offset);
      } else if (offset + 12 <= size) {
         *(global uint3 *)(dst_ptr + offset) = *(global uint3 *)(src_ptr + offset);
      } else if (offset + 8 <= size) {
         *(global uint2 *)(dst_ptr + offset) = *(global uint2 *)(src_ptr + offset);
      } else if (offset + 4 <= size) {
         *(global uint *)(dst_ptr + offset) = *(global uint *)(src_ptr + offset);
      }
   }
}

/* Copy size from src_ptr to dst_ptr for using a single lane with size
 * multiple of 4.
 */
void genX(set_data)(global void *dst_ptr,
                    uint32_t size,
                    uint32_t data)
{
   for (uint32_t offset = 0; offset < size; offset += 16) {
      if (offset + 16 <= size) {
         *(global uint4 *)(dst_ptr + offset) = (uint4)(data);
      } else if (offset + 12 <= size) {
         *(global uint3 *)(dst_ptr + offset) = (uint3)(data);
      } else if (offset + 8 <= size) {
         *(global uint2 *)(dst_ptr + offset) = (uint2)(data);
      } else if (offset + 4 <= size) {
         *(global uint *)(dst_ptr + offset) = data;
      }
   }
}

#define intel_unpack(src, T, name)                      \
   struct GENX(T) name;                                 \
   GENX(T##_unpack)((CONST uint8_t *)(src), &name)

#include "vulkan/anv_types.h"

uint4
genX(resinfo_indirect)(CONST void *descriptor, uint32_t lod,
                       unsigned nr_comps, enum glsl_sampler_dim dim,
                       bool is_array)
{
   /* This code should only be used pre Gfx12.5 */
   CONST struct anv_image_descriptor *desc = (CONST struct anv_image_descriptor *)descriptor;

   /* From the Vulkan spec:
    *
    *    OpImageQuery*... return 0 if the bound descriptor is a null
    *    descriptor
    */
   if (desc->address == 0)
      return (uint4)(0, 0, 0, 0);

   if (dim == GLSL_SAMPLER_DIM_BUF)
      return (uint4)(desc->width, 0, 0, 1);

   uint4 size = (uint4)(desc->width, desc->height, desc->depth_or_layer_count, desc->level_count - lod);
   lod += desc->base_mip_level;

   /* 1D Arrays have their second component as the layer count */
   if (dim == GLSL_SAMPLER_DIM_1D && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* VK_EXT_image_sliced_view_of_3d */
   if (dim == GLSL_SAMPLER_DIM_3D)
      size.z = MIN2(desc->z_slice_count, size.z);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      size.y = size.x;

   return size;
}

uint4
genX(resinfo)(CONST void *descriptor, uint32_t lod,
              unsigned nr_comps, enum glsl_sampler_dim dim,
              bool is_array)
{
   intel_unpack(descriptor, RENDER_SURFACE_STATE, ss);

   /* From the Vulkan spec:
    *
    *    OpImageQuery*... return 0 if the bound descriptor is a null
    *    descriptor
    */
   if (ss.SurfaceType == SURFTYPE_NULL)
      return (uint4)(0, 0, 0, 0);

   /* Buffer length is encoded in the descriptor's Width/Height/Depth
    * fields.
    */
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      return (uint4)((ss.Width | (ss.Height << 7) | (ss.Depth << 21)) + 1,
                     0, 0, 1);
   }

   /* Load standard dimensions */
   uint4 size = (uint4)(ss.Width + 1, ss.Height + 1, ss.Depth + 1, ss.MIPCountLOD + 1 - lod);

   lod += ss.SurfaceMinLOD;

   /* 1D Arrays have their second component as the layer count */
   if (dim == GLSL_SAMPLER_DIM_1D && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* VK_EXT_image_sliced_view_of_3d */
   if (dim == GLSL_SAMPLER_DIM_3D)
      size.z = MIN2(ss.RenderTargetViewExtent + 1, size.z);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      size.y = size.x;

   return size;
}
