/*
 * Copyright © 2021 Red Hat
 * Copyright © 2021 Valve Corporation
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
 *
 * Authors:
 *    Dave Airlie <airlied@redhat.com>
 *    Mike Blumenkrantz <michael.blumenkrantz@gmail.com>
 */

#ifndef ZINK_VIDEO_H
#define ZINK_VIDEO_H

#include "pipe/p_video_codec.h"
#include "util/u_upload_mgr.h"
#include "zink_types.h"

struct zink_video_codec {
   struct pipe_video_codec base;
   struct pipe_screen *screen;
   uint8_t *bs_ptr;
   size_t bs_size;
   unsigned bs_alloc_size;
   unsigned bs_offset;
   bool error;
   struct u_upload_mgr *bitstream_mgr;
   struct zink_resource *bitstream_res;
   VkVideoSessionParametersKHR params;
   unsigned num_priv_mems;
   unsigned max_dpb_slots;
   struct zink_bo **priv_mems;
   VkVideoSessionKHR session;
   bool reset_sent;
   bool coincide_dpb;
   bool dpb_array;
   size_t srcbuf_align;
   size_t srcbuf_offset_align;
   struct pipe_video_buffer *render_pic_list[17];
   /* the last decoded picture, to find the second field of a pair */
   struct pipe_video_buffer *last_target;
   uint32_t last_frame_num;
   bool last_field;
   bool last_bottom;
   /* for separate dpb/dst storage */
   struct pipe_resource *dpb_res[17];
   VkVideoPictureResourceInfoKHR dpb_resources[17];
};

struct zink_video_profile {
   VkVideoDecodeH264ProfileInfoKHR h264;
   VkVideoProfileInfoKHR profile;
};

struct zink_video_profile_info {
   struct zink_video_profile h264;
   VkVideoProfileInfoKHR profiles[4];
   VkVideoProfileListInfoKHR list;
};

void
zink_video_fill_profiles(struct zink_screen *screen,
			 struct zink_video_profile_info *profiles,
                         enum pipe_video_profile profile, uint32_t luma_depth);

struct zink_video_caps_info {
   VkVideoCapabilitiesKHR caps;
   VkVideoDecodeCapabilitiesKHR dec_caps;
   VkVideoDecodeH264CapabilitiesKHR h264_dec_caps;
};

bool
zink_video_fill_caps(struct zink_screen *screen,
                     enum pipe_video_profile profile,
                     enum pipe_video_entrypoint entrypoint,
                     bool bit_depth_10,
                     struct zink_video_caps_info *caps_info);
bool
zink_video_fill_single_profile(struct zink_screen *screen,
                               enum pipe_video_profile profile,
                               uint32_t luma_depth,
                               struct zink_video_profile *out_prof);
void
zink_video_init(struct zink_context *ctx);

static inline uint32_t
zink_video_get_format_bit_depth(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_NV12:
      return 8;
   case PIPE_FORMAT_P010:
      return 10;
   default:
      return 0;
   }
}

void
zink_video_init_format_props(struct zink_screen *screen);

VkResult
zink_fill_video_format_props(struct zink_screen *screen,
                             VkImageUsageFlags usage,
                             enum pipe_video_profile profile,
                             uint32_t bit_depth,
                             struct zink_video_format_prop *props);

#endif
