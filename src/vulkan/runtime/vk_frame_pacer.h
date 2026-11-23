/*
 * Copyright © 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef VK_FRAME_PACER_H
#define VK_FRAME_PACER_H

#include "vk_object.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file vk_frame_pacer.h
 *
 * vk_frame_pacer:
 *
 * This framework provides basic utility for the implementation of low latency
 * frame pacing extensions.
 *
 * The functionality includes the measurement of frame times where frames
 * consist of all submissions between two calls to vkQueuePresentKHR as well
 * as the measurement of queuing delays.
 * For this purpose, calibrated timestamps are taken at submission time as well
 * as at the begin of each submission execution on the device.
 *
 * Usage:
 * - implement a callback function that updates user data with the provided
 *   vk_frame_time_info.
 * - on vkCreateDevice: call vk_frame_pacer_init()
 * - on VkDestroyDevice: call vk_frame_pacer_finish()
 *
 * - on vkLatencySleepNV / vkAntiLagUpdateAMD:
 *   // critical section begin
 *   {
 *      vk_frame_pacer_enable/disable(); // based on mode
 *
 *      // Update user data with newest vk_frame_time_info
 *      vk_frame_pacer_evaluate();
 *
 *      // calculate next deadline based on updated user data
 *      user_data = vk_frame_pacer_get_user_data()
 *      deadline = calculate_deadline(user_data)
 *
 *      // Recalibrate occassionally to avoid drifting.
 *      if (deadline > recalibration_deadline)
 *         vk_frame_pacer_calibrate();
 *
 *      os_time_nanosleep_until(deadline);
 *   }
 *   // critical section end
 *
 */

struct vk_frame_pacer;

struct vk_frame_time_info {
   /* Frame time as measured from before the first submission until after the
    * last submission, associated with a frame.
    */
   uint64_t frame_time;
   /* The minimum queuing-delay as measured between QueueSubmit2 and the
    * execution of the submission on the GPU.
    */
   int64_t min_delay;
};

/*
 * Callback function for each evaluated frame from vk_frame_pacer_evaluate.
 *
 * This function is being called with the most recent vk_frame_time_info and
 * should update the provided user data with the necessary information.
 */
typedef void (*vk_frame_time_cb)(const struct vk_frame_time_info *, void *);

/*
 * Initialize vk_frame_pacer (during vkCreateDevice).
 *
 * This function allocates all necessary memory for the frame_pacer upfront.
 *
 * Note: This function MUST only be called after vk_device is fully initialized,
 *       including all vk_queues and the dispatch table.
 *
 * Note: This function updates QueueSubmit2 and QueuePresentKHR from the
 *       device dispatch_table.
 */
VkResult vk_frame_pacer_init(struct vk_device *device, vk_frame_time_cb update_cb,
                             size_t data_size);

/*
 * De-initialize vk_frame_pacer (during vkDestroyDevice).
 *
 * Note: This function MUST be called before destroying vk_device.
 */
void vk_frame_pacer_finish(struct vk_device *device);

/*
 * Enable frame pacer.
 *
 * When enabled, the frame pacer adds timestamp queries two all queue
 * submissions between each two calls to QueuePresentKHR.
 *
 * When called for the first time, this function creates one CommandPool, one
 * QueryPool, a timeline Semaphore, and a number of CommandBuffers per Queue
 * which are being used for timestamp queries.
 *
 * Repeated calls have no effect.
 *
 * Note: This function is not thread-safe against concurrent calls.
 */
void vk_frame_pacer_enable(struct vk_frame_pacer *frame_pacer);

/*
 * Disable frame pacer.
 *
 * Repeated calls have no effect.
 *
 * Note: This function is not thread-safe against concurrent calls.
 */
void vk_frame_pacer_disable(struct vk_frame_pacer *frame_pacer);

/*
 * Evaluate any already completed frames.
 *
 * After evaluation, this function populates vk_frame_time_info and calls the
 * provided callback function.
 *
 * Note: This function is not thread-safe against concurrent calls.
 */
void vk_frame_pacer_evaluate(struct vk_frame_pacer *frame_pacer);

/*
 * Calibrate device and host timestamps.
 * This function should be called occassionally.
 */
void vk_frame_pacer_calibrate(struct vk_frame_pacer *frame_pacer);

/*
 * Returns a pointer to the user data of size data_size.
 */
void *vk_frame_pacer_get_user_data(struct vk_frame_pacer *frame_pacer);

#ifdef __cplusplus
}
#endif

#endif /* VK_FRAME_PACER_H */
