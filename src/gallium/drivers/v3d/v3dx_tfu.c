/*
 * Copyright © 2022 Raspberry Pi Ltd
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

#include "v3d_context.h"
#include "broadcom/common/v3d_tfu.h"
#include "util/perf/cpu_trace.h"
#include "v3dx_format_table.h"

bool
v3dX(tfu)(struct pipe_context *pctx,
          struct pipe_resource *pdst,
          struct pipe_resource *psrc,
          unsigned int src_level,
          unsigned int base_level,
          unsigned int last_level,
          unsigned int src_layer,
          unsigned int dst_layer,
          bool for_mipmap)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_screen *screen = v3d->screen;
        struct v3d_resource *src = v3d_resource(psrc);
        struct v3d_resource *dst = v3d_resource(pdst);
        struct v3d_resource_slice *src_base_slice = &src->slices[src_level];
        struct v3d_resource_slice *dst_base_slice = &dst->slices[base_level];
        int msaa_scale = pdst->nr_samples > 1 ? 2 : 1;
        int width = u_minify(pdst->width0, base_level) * msaa_scale;
        int height = u_minify(pdst->height0, base_level) * msaa_scale;
        enum pipe_format pformat;

        if (psrc->format != pdst->format)
                return false;
        if (psrc->nr_samples != pdst->nr_samples)
                return false;

        assert(!for_mipmap || pdst->target != PIPE_TEXTURE_3D);

#if V3D_VERSION == 42
        /* Can't write to raster. */
        if (dst_base_slice->tiling == V3D_TILING_RASTER)
                return false;
#endif

        /* TFU requires UIF when generating mipmaps */
        if (for_mipmap && dst_base_slice->tiling == V3D_TILING_RASTER)
                return false;

        /* When using TFU for blit, we are doing exact copies (both input and
         * output format must be the same, no scaling, etc), so there is no
         * pixel format conversions. Thus we can rewrite the format to use one
         * that is TFU compatible based on its texel size.
         */
        if (for_mipmap) {
                pformat = pdst->format;
        } else {
                switch (dst->cpp) {
                case 16: pformat = PIPE_FORMAT_R32G32B32A32_FLOAT;   break;
                case 8:  pformat = PIPE_FORMAT_R16G16B16A16_FLOAT;   break;
                case 4:  pformat = PIPE_FORMAT_R32_FLOAT;            break;
                case 2:  pformat = PIPE_FORMAT_R16_FLOAT;            break;
                case 1:  pformat = PIPE_FORMAT_R8_UNORM;             break;
                default: UNREACHABLE("unsupported format bit-size"); break;
                };
        }

        enum V3DX(Texture_Data_Formats) tex_format = v3d_get_tex_format(&screen->devinfo, pformat);

        if (!v3dX(tfu_supports_tex_format)(tex_format, for_mipmap)) {
                assert(for_mipmap);
                return false;
        }
        MESA_TRACE_FUNC();

        v3d_flush_jobs_writing_resource(v3d, psrc, V3D_FLUSH_DEFAULT, false);
        v3d_flush_jobs_reading_resource(v3d, pdst, V3D_FLUSH_DEFAULT, false);

        struct drm_v3d_submit_tfu tfu = {
                .ios = (height << 16) | width,
                .bo_handles = {
                        dst->bo->handle,
                        src != dst ? src->bo->handle : 0
                },
                .in_sync = v3d->out_sync,
                .out_sync = v3d->out_sync,
        };
        uint32_t src_offset = (src->bo->offset +
                               v3d_layer_offset(psrc, src_level, src_layer));
        tfu.iia |= src_offset;

        uint32_t dst_offset = (dst->bo->offset +
                               v3d_layer_offset(pdst, base_level, dst_layer));
        tfu.ioa |= dst_offset;

        switch (src_base_slice->tiling) {
        case V3D_TILING_UIF_NO_XOR:
        case V3D_TILING_UIF_XOR:
                tfu.iis |= (src_base_slice->padded_height /
                            (2 * v3d_utile_height(src->cpp)));
                break;
        case V3D_TILING_RASTER:
                tfu.iis |= src_base_slice->stride / src->cpp;
                break;
        case V3D_TILING_LINEARTILE:
        case V3D_TILING_UBLINEAR_1_COLUMN:
        case V3D_TILING_UBLINEAR_2_COLUMN:
                break;
       }

#if V3D_VERSION == 42
        if (src_base_slice->tiling == V3D_TILING_RASTER) {
                tfu.icfg |= (V3D33_TFU_ICFG_FORMAT_RASTER <<
                             V3D33_TFU_ICFG_FORMAT_SHIFT);
        } else {
                tfu.icfg |= ((V3D33_TFU_ICFG_FORMAT_LINEARTILE +
                              (src_base_slice->tiling - V3D_TILING_LINEARTILE)) <<
                             V3D33_TFU_ICFG_FORMAT_SHIFT);
        }
        tfu.icfg |= tex_format << V3D33_TFU_ICFG_TTYPE_SHIFT;

        if (last_level != base_level)
                tfu.ioa |= V3D33_TFU_IOA_DIMTW;

        tfu.ioa |= ((V3D33_TFU_IOA_FORMAT_LINEARTILE +
                     (dst_base_slice->tiling - V3D_TILING_LINEARTILE)) <<
                    V3D33_TFU_IOA_FORMAT_SHIFT);

        tfu.icfg |= (last_level - base_level) << V3D33_TFU_ICFG_NUMMM_SHIFT;

        /* If we're writing level 0 (!IOA_DIMTW), then we need to supply the
         * OPAD field for the destination (how many extra UIF blocks beyond
         * those necessary to cover the height).  When filling mipmaps, the
         * miplevel 1+ tiling state is inferred.
         */
        if (dst_base_slice->tiling == V3D_TILING_UIF_NO_XOR ||
            dst_base_slice->tiling == V3D_TILING_UIF_XOR) {
                int uif_block_h = 2 * v3d_utile_height(dst->cpp);
                int implicit_padded_height = align(height, uif_block_h);

                tfu.icfg |= (((dst_base_slice->padded_height -
                               implicit_padded_height) / uif_block_h) <<
                             V3D33_TFU_ICFG_OPAD_SHIFT);
        }
#endif /* V3D_VERSION == 42 */

#if V3D_VERSION >= 71
        if (src_base_slice->tiling == V3D_TILING_RASTER) {
                tfu.icfg = V3D71_TFU_ICFG_FORMAT_RASTER << V3D71_TFU_ICFG_IFORMAT_SHIFT;
        } else {
                tfu.icfg = (V3D71_TFU_ICFG_FORMAT_LINEARTILE +
                            (src_base_slice->tiling - V3D_TILING_LINEARTILE)) <<
                        V3D71_TFU_ICFG_IFORMAT_SHIFT;
        }
        tfu.icfg |= tex_format << V3D71_TFU_ICFG_OTYPE_SHIFT;

        if (last_level != base_level)
                tfu.v71.ioc |= V3D71_TFU_IOC_DIMTW;

        if (dst_base_slice->tiling == V3D_TILING_RASTER) {
                tfu.v71.ioc |= V3D71_TFU_IOC_FORMAT_RASTER << V3D71_TFU_IOC_FORMAT_SHIFT;
        } else {
                tfu.v71.ioc |= ((V3D71_TFU_IOC_FORMAT_LINEARTILE +
                                 (dst_base_slice->tiling - V3D_TILING_LINEARTILE)) <<
                                V3D71_TFU_IOC_FORMAT_SHIFT);
        }

        switch (dst_base_slice->tiling) {
        case V3D_TILING_UIF_NO_XOR:
        case V3D_TILING_UIF_XOR:
                tfu.v71.ioc |=
                        (dst_base_slice->padded_height / (2 * v3d_utile_height(dst->cpp))) <<
                        V3D71_TFU_IOC_STRIDE_SHIFT;
                break;
        case V3D_TILING_RASTER:
                tfu.v71.ioc |= (dst_base_slice->stride / dst->cpp) <<
                        V3D71_TFU_IOC_STRIDE_SHIFT;
                break;
        default:
                break;
        }

        tfu.v71.ioc |= (last_level - base_level) << V3D71_TFU_IOC_NUMMM_SHIFT;
#endif /* V3D_VERSION >= 71*/

        int ret = v3d_ioctl(screen->fd, DRM_IOCTL_V3D_SUBMIT_TFU, &tfu);
        if (ret != 0) {
                mesa_loge("Failed to submit TFU job: %d", ret);
                return false;
        }
        if (V3D_DBG(SYNC)) {
                drmSyncobjWait(v3d->fd, &v3d->out_sync, 1, INT64_MAX,
                               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
        }

        dst->writes++;

        return true;
}

bool
v3dX(tfu_sand)(struct pipe_context *pctx,
               struct pipe_resource *pdst,
               unsigned int dst_level,
               struct pipe_resource *psrc,
               unsigned int src_level)
{
#if V3D_VERSION >= 71
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_screen *screen = v3d->screen;
        struct v3d_resource *src = v3d_resource(psrc);
        struct v3d_resource *dst = v3d_resource(pdst);
        struct v3d_resource_slice *dst_slice = &dst->slices[dst_level];
        uint32_t width = u_minify(pdst->width0, dst_level);
        uint32_t height = u_minify(pdst->height0, dst_level);
        enum V3DX(Texture_Data_Formats) tex_format;
        uint32_t icfmt;

        assert(psrc->format == pdst->format);

        switch (psrc->format) {
        case PIPE_FORMAT_R8_UNORM:
        case PIPE_FORMAT_R8G8_UNORM:
                tex_format = v3d_get_tex_format(&screen->devinfo, psrc->format);
                icfmt = V3D71_TFU_ICFG_ICFMT_8BIT;
                break;
        /* P030 SAND planes arrive as R16/R16G16 here as V3D doesn't support
         * sampling P030 directly. For a 10-bit SAND input, the TFU only
         * allows R10x6/RG10x6 as its output type.
         */
        case PIPE_FORMAT_R16_UNORM:
                tex_format = TEXTURE_DATA_FORMAT_R10X6;
                icfmt = V3D71_TFU_ICFG_ICFMT_101010_2;
                break;
        case PIPE_FORMAT_R16G16_UNORM:
                tex_format = TEXTURE_DATA_FORMAT_RG10X6;
                icfmt = V3D71_TFU_ICFG_ICFMT_101010_2;
                break;
        default:
                return false;
        }

        assert(dst_slice->tiling != V3D_TILING_RASTER);

        /* With Sand Video input, image stride should be in multiple of 8. */
        if (src->sand_col128_stride % 8 != 0 || src->sand_col128_stride > 0x7fff)
                return false;

        /* With Sand Video input, the source memory address must be 1KB aligned. */
        uint32_t src_offset = src->bo->offset + v3d_layer_offset(psrc, src_level, 0);
        if (src_offset % 1024 != 0)
                return false;

        MESA_TRACE_FUNC();

        v3d_flush_jobs_writing_resource(v3d, psrc, V3D_FLUSH_DEFAULT, false);
        v3d_flush_jobs_reading_resource(v3d, pdst, V3D_FLUSH_DEFAULT, false);

        struct drm_v3d_submit_tfu tfu = {
                .iia = src_offset,
                .iis = src->sand_col128_stride,
                .ioa = dst->bo->offset + v3d_layer_offset(pdst, dst_level, 0),
                .ios = (height << 16) | width,
                .bo_handles = {
                        dst->bo->handle,
                        src != dst ? src->bo->handle : 0
                },
                .in_sync = v3d->out_sync,
                .out_sync = v3d->out_sync,
        };

        tfu.icfg = (V3D71_TFU_ICFG_SVSWIZ_NONE << V3D71_TFU_ICFG_SVSWIZ_SHIFT) |
                   (V3D71_TFU_ICFG_SVTWID_128 << V3D71_TFU_ICFG_SVTWID_SHIFT) |
                   (V3D71_TFU_ICFG_FORMAT_SAND << V3D71_TFU_ICFG_IFORMAT_SHIFT) |
                   (tex_format << V3D71_TFU_ICFG_OTYPE_SHIFT) |
                   (icfmt << V3D71_TFU_ICFG_ICFMT_SHIFT);

        tfu.v71.ioc = (V3D71_TFU_IOC_FORMAT_LINEARTILE +
                       (dst_slice->tiling - V3D_TILING_LINEARTILE)) <<
                      V3D71_TFU_IOC_FORMAT_SHIFT;

        if (dst_slice->tiling == V3D_TILING_UIF_NO_XOR ||
            dst_slice->tiling == V3D_TILING_UIF_XOR) {
                tfu.v71.ioc |=
                        (dst_slice->padded_height / (2 * v3d_utile_height(dst->cpp))) <<
                        V3D71_TFU_IOC_STRIDE_SHIFT;
        }

        int ret = v3d_ioctl(screen->fd, DRM_IOCTL_V3D_SUBMIT_TFU, &tfu);
        if (ret != 0) {
                mesa_loge("Failed to submit TFU job: %d", ret);
                return false;
        }
        if (V3D_DBG(SYNC)) {
                drmSyncobjWait(v3d->fd, &v3d->out_sync, 1, INT64_MAX,
                               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
        }

        dst->writes++;

        return true;
#else
        return false;
#endif
}

bool
v3dX(tfu_clear_buffer)(struct pipe_context *pctx,
                       struct pipe_resource *prsc,
                       unsigned offset,
                       unsigned size,
                       const void *clear_value,
                       int clear_value_size)
{
#if V3D_VERSION < 71
        return false;
#else
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_screen *screen = v3d->screen;
        struct v3d_resource *rsc = v3d_resource(prsc);

        if (!screen->has_tfu_ccol)
                return false;

        uint32_t cpp = clear_value_size;
        uint32_t ccol[4] = { 0 };

        if (size == 0)
                return true;

        if (cpp > 16 || !util_is_power_of_two_nonzero(cpp))
                return false;

        /* For cpp = 1 or cpp = 2, repeat the pattern every 1 or 2 bytes and
         * use cpp = 4 when possible. This allows us to take 4x (or 2x) fewer
         * pixels, making it more likely for clears to become single rows or
         * generate fewer TFU jobs.
         */
        if (cpp < 4 && offset % 4 == 0 && size % 4 == 0) {
                for (uint32_t i = 0; i < 4; i += cpp)
                        memcpy((uint8_t *)ccol + i, clear_value, cpp);
                cpp = 4;
        } else {
                memcpy(ccol, clear_value, cpp);
        }

        enum pipe_format pformat;
        switch (cpp) {
        case 16: pformat = PIPE_FORMAT_R32G32B32A32_FLOAT; break;
        case 8:  pformat = PIPE_FORMAT_R16G16B16A16_UNORM; break;
        case 4:  pformat = PIPE_FORMAT_R8G8B8A8_UNORM;     break;
        case 2:  pformat = PIPE_FORMAT_R16_UNORM;          break;
        default: pformat = PIPE_FORMAT_R8_UNORM;           break;
        }

        enum V3DX(Texture_Data_Formats) tex_format = v3d_get_tex_format(&screen->devinfo, pformat);
        assert(v3dX(tfu_supports_tex_format)(tex_format, false));

        MESA_TRACE_FUNC();

        v3d_flush_jobs_reading_resource(v3d, prsc, V3D_FLUSH_DEFAULT, false);

        const uint32_t icfg =
                (V3D71_TFU_ICFG_FORMAT_CONSTANT_COLOUR << V3D71_TFU_ICFG_IFORMAT_SHIFT) |
                (tex_format << V3D71_TFU_ICFG_OTYPE_SHIFT);

        uint32_t remaining = size / cpp;
        while (remaining > 0) {
                uint32_t width = v3d_tfu_raster_job_width(remaining, cpp);
                uint32_t height = MAX2(1, MIN2(remaining / width, V3D_TFU_MAX_DIM));
                uint32_t pixels = width * height;
                assert(pixels <= remaining);

                struct drm_v3d_submit_tfu tfu = {
                        .icfg = icfg,
                        .ioa = rsc->bo->offset + offset,
                        .ios = (height << 16) | width,
                        .bo_handles = { rsc->bo->handle },
                        .in_sync = v3d->out_sync,
                        .out_sync = v3d->out_sync,
                        .v71 = {
                                .ioc = (V3D71_TFU_IOC_FORMAT_RASTER << V3D71_TFU_IOC_FORMAT_SHIFT) |
                                       (width << V3D71_TFU_IOC_STRIDE_SHIFT),
                                .ccol = { ccol[0], ccol[1], ccol[2], ccol[3] },
                        },
                };

                int ret = v3d_ioctl(screen->fd, DRM_IOCTL_V3D_SUBMIT_TFU, &tfu);
                if (ret != 0) {
                        mesa_loge("Failed to submit TFU clear job: %d", ret);
                        return false;
                }

                remaining -= pixels;
                offset += pixels * cpp;
        }

        if (V3D_DBG(SYNC)) {
                drmSyncobjWait(v3d->fd, &v3d->out_sync, 1, INT64_MAX,
                               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
        }

        rsc->writes++;

        return true;
#endif
}
