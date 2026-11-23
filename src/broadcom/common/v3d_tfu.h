/*
 * Copyright © 2021 Raspberry Pi Ltd
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

#ifndef V3D_TFU_H
#define V3D_TFU_H

#include <stdint.h>

#include "util/macros.h"

/* Maximum width and height (in pixels) per submitted TFU job. The ios
 * register encodes width and height as 14-bit values each.
 */
#define V3D_TFU_MAX_DIM 16384

/* Disable level 0 write, just write following mipmaps */
#define V3D33_TFU_IOA_DIMTW (1 << 0)
#define V3D33_TFU_IOA_FORMAT_SHIFT 3
#define V3D33_TFU_IOA_FORMAT_LINEARTILE 3
#define V3D33_TFU_IOA_FORMAT_UBLINEAR_1_COLUMN 4
#define V3D33_TFU_IOA_FORMAT_UBLINEAR_2_COLUMN 5
#define V3D33_TFU_IOA_FORMAT_UIF_NO_XOR 6
#define V3D33_TFU_IOA_FORMAT_UIF_XOR 7

#define V3D33_TFU_ICFG_NUMMM_SHIFT 5
#define V3D33_TFU_ICFG_TTYPE_SHIFT 9

#define V3D33_TFU_ICFG_OPAD_SHIFT 22

#define V3D33_TFU_ICFG_FORMAT_SHIFT 18
#define V3D33_TFU_ICFG_FORMAT_RASTER 0
#define V3D33_TFU_ICFG_FORMAT_SAND_128 1
#define V3D33_TFU_ICFG_FORMAT_SAND_256 2
#define V3D33_TFU_ICFG_FORMAT_LINEARTILE 11
#define V3D33_TFU_ICFG_FORMAT_UBLINEAR_1_COLUMN 12
#define V3D33_TFU_ICFG_FORMAT_UBLINEAR_2_COLUMN 13
#define V3D33_TFU_ICFG_FORMAT_UIF_NO_XOR 14
#define V3D33_TFU_ICFG_FORMAT_UIF_XOR 15

/* Disable level 0 write, just write following mipmaps */
#define V3D71_TFU_IOC_DIMTW (1 << 0)
#define V3D71_TFU_IOC_FORMAT_SHIFT              12
#define V3D71_TFU_IOC_FORMAT_RASTER              0
#define V3D71_TFU_IOC_FORMAT_LINEARTILE          3
#define V3D71_TFU_IOC_FORMAT_UBLINEAR_1_COLUMN   4
#define V3D71_TFU_IOC_FORMAT_UBLINEAR_2_COLUMN   5
#define V3D71_TFU_IOC_FORMAT_UIF_NO_XOR          6
#define V3D71_TFU_IOC_FORMAT_UIF_XOR             7

#define V3D71_TFU_IOC_STRIDE_SHIFT              16
#define V3D71_TFU_IOC_NUMMM_SHIFT                4

#define V3D71_TFU_ICFG_OTYPE_SHIFT              16
#define V3D71_TFU_ICFG_IFORMAT_SHIFT            23
#define V3D71_TFU_ICFG_FORMAT_RASTER             0
#define V3D71_TFU_ICFG_FORMAT_SAND               1
#define V3D71_TFU_ICFG_FORMAT_CONSTANT_COLOUR    2
#define V3D71_TFU_ICFG_FORMAT_LINEARTILE         3
#define V3D71_TFU_ICFG_FORMAT_UBLINEAR_1_COLUMN  4
#define V3D71_TFU_ICFG_FORMAT_UBLINEAR_2_COLUMN  5
#define V3D71_TFU_ICFG_FORMAT_UIF_NO_XOR         6
#define V3D71_TFU_ICFG_FORMAT_UIF_XOR            7

#define V3D71_TFU_ICFG_SVSWIZ_SHIFT             28
#define V3D71_TFU_ICFG_SVSWIZ_NONE              15

#define V3D71_TFU_ICFG_SVTWID_SHIFT             26
#define V3D71_TFU_ICFG_SVTWID_128                0

#define V3D71_TFU_ICFG_ICFMT_SHIFT               8
#define V3D71_TFU_ICFG_ICFMT_8BIT                1
#define V3D71_TFU_ICFG_ICFMT_101010_2            3

/* Limits used to shape TFU jobs for buffer fills and copies.
 *
 * These limits are not defined by the hardware. They were selected after
 * analyzing the memory throughput of buffer fills and copies for different
 * width and height sizes.
 */
#define V3D_TFU_RASTER_SINGLE_ROW_MAX 1024
#define V3D_TFU_RASTER_ROW_MAX_PIXELS 256
#define V3D_TFU_RASTER_ROW_MAX_BYTES 1024
#define V3D_TFU_RASTER_ROW_MIN_PIXELS 64

/* Uses a divisor search to find the width of the raster TFU job. The
 * algorithm aims to avoid an extra remainder job for sizes that aren't
 * a multiple of the maximum row width.
 */
static inline uint32_t
v3d_tfu_raster_job_width(uint32_t pixels, uint32_t cpp)
{
   if (pixels <= V3D_TFU_RASTER_SINGLE_ROW_MAX)
      return pixels;

   /* Rows are kept at most V3D_TFU_RASTER_ROW_MAX_PIXELS pixels and at
    * most V3D_TFU_RASTER_ROW_MAX_BYTES bytes wide (caps the stride of
    * wide pixels, such as 8 or 16 bytes).
    */
   const uint32_t max_width =
      MIN2(V3D_TFU_RASTER_ROW_MAX_PIXELS, V3D_TFU_RASTER_ROW_MAX_BYTES / cpp);

   if (pixels % max_width == 0 || pixels >= V3D_TFU_MAX_DIM)
      return max_width;

   /* Prefer a narrower width that divides the pixel count, so the whole
    * operation is placed in a single job.
    */
   for (uint32_t w = max_width - 1; w >= V3D_TFU_RASTER_ROW_MIN_PIXELS; w--) {
      if (pixels % w == 0)
         return w;
   }

   return max_width;
}

#endif
