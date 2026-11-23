/*
 * Copyright © 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/simple_mtx.h"
#include "vulkan/vulkan_core.h"
#include "radv_device.h"
#include "radv_entrypoints.h"
#include "vk_frame_pacer.h"

#define ONE_MS_IN_NS INT64_C(1000000)

struct anti_lag_data {
   simple_mtx_t mtx;
   uint64_t avg_frame_time;
   uint64_t deviation;
   uint64_t prev_input_begin;
   int64_t queuing_delay;
   uint64_t recalibrate_when;
};

static void
frame_time_cb(const struct vk_frame_time_info *info, void *_)
{
   struct anti_lag_data *data = _;
   if (info->frame_time) {
      /* Calculate average absolute deviation as exponential moving average with alpha = 0.125.
       * Total frame time being time from first QueueSubmit is made until all GPU work is drained
       * on the GPU.
       */
      int64_t expected_total_frame_time = data->queuing_delay + data->avg_frame_time;
      int64_t current_total_frame_time = info->min_delay + info->frame_time;
      int64_t deviation = expected_total_frame_time - current_total_frame_time;
      int64_t diff = (deviation < 0 ? -deviation : deviation) - data->deviation;
      data->deviation += diff / 8;

      /* Update frame time: Use an exponential moving average with alpha = 0.5. */
      data->avg_frame_time = (data->avg_frame_time + info->frame_time) / 2;
   }

   /* Update queuing delay: Discount the previous delay and add half of the frame's minimum delay.
    * This formula was decided upon after trial and error testing:
    * The rational is that a longer frame-time only materializes after multiple frames as queuing
    * delay. If we then delay the next frame, we still receive increased queuing delays from the
    * following callbacks. But since we already accounted for it, any further addition would lead
    * to underpacing. So the idea here is basically to be able to react quickly to sudden changes
    * without ping-ponging between extremes. This might be subject to future changes.
    */
   data->queuing_delay = -data->queuing_delay / 4 + info->min_delay / 2;
}

VkResult
radv_device_init_anti_lag(struct radv_device *device)
{
   VkResult result = vk_frame_pacer_init(&device->vk, frame_time_cb, sizeof(struct anti_lag_data));

   if (result == VK_SUCCESS) {
      struct anti_lag_data *data = vk_frame_pacer_get_user_data(device->vk.frame_pacer);
      simple_mtx_init(&data->mtx, mtx_plain);
   }

   return result;
}

void
radv_device_finish_anti_lag(struct radv_device *device)
{
   if (device->vk.frame_pacer) {
      struct anti_lag_data *data = vk_frame_pacer_get_user_data(device->vk.frame_pacer);
      simple_mtx_destroy(&data->mtx);
      vk_frame_pacer_finish(&device->vk);
   }
}

static uint64_t
get_wait_time(struct anti_lag_data *data)
{
   /* Start the next input sampling after the period of one frame.
    * We add the discounted queuing delay as that's what we try to minimize.
    * Subtract the average frame time deviation as slack time, so that
    * we don't drain the GPU from work.
    */
   int64_t delay = data->avg_frame_time + data->queuing_delay - MIN2(data->deviation, ONE_MS_IN_NS);
   return MAX2(delay, 0);
}

VKAPI_ATTR void VKAPI_CALL
radv_AntiLagUpdateAMD(VkDevice _device, const VkAntiLagDataAMD *pData)
{
   if (pData == NULL)
      return;

   VK_FROM_HANDLE(vk_device, device, _device);
   struct vk_frame_pacer *frame_pacer = device->frame_pacer;
   struct anti_lag_data *data = vk_frame_pacer_get_user_data(frame_pacer);

   /* Lock this function, in order to avoid race conditions on frame evaluation. */
   simple_mtx_lock(&data->mtx);

   if (pData->mode == VK_ANTI_LAG_MODE_OFF_AMD) {
      /* Application request to disable Anti-Lag. */
      vk_frame_pacer_disable(frame_pacer);
      simple_mtx_unlock(&data->mtx);
      return;
   }

   vk_frame_pacer_enable(frame_pacer);

   if (pData->pPresentationInfo) {
      /* The same frameIndex value should be used with VK_ANTI_LAG_STAGE_INPUT_AMD before
       * the frame begins and with VK_ANTI_LAG_STAGE_PRESENT_AMD when the frame ends.
       */
      UNUSED uint64_t frame_idx = pData->pPresentationInfo->frameIndex;

      /* This marks the end of the input stage of the current frame. */
      if (pData->pPresentationInfo->stage == VK_ANTI_LAG_STAGE_PRESENT_AMD) {
         /* TODO: If available, this can be used to calculate the simulation time. */
         simple_mtx_unlock(&data->mtx);
         return;
      }
   }

   /* VK_ANTI_LAG_STAGE_INPUT_AMD: This marks the begin of a new frame.
    * Evaluate previous frames in order to determine the wait time.
    */
   vk_frame_pacer_evaluate(frame_pacer);

   uint64_t delay = get_wait_time(data);

   /* Ensure maxFPS adherence. */
   if (pData->maxFPS) {
      uint64_t frametime_period = ONE_SECOND_IN_NS / pData->maxFPS;
      delay = MAX2(delay, frametime_period);
   }

   uint64_t next_deadline = data->prev_input_begin + delay;
   uint64_t now = os_time_get_nano();

   /* Recalibrate every now and then. */
   if (next_deadline > data->recalibrate_when) {
      vk_frame_pacer_calibrate(frame_pacer);
      /* Take a new calibrated timestamp every 8 seconds. */
      data->recalibrate_when = next_deadline + 8 * ONE_SECOND_IN_NS;
   }

   data->prev_input_begin = MAX2(now, next_deadline);

   simple_mtx_unlock(&data->mtx);

   /* Sleep until deadline is met. */
   os_time_nanosleep_until(next_deadline);
}