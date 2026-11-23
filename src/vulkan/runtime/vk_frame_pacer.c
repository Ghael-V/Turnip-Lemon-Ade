/*
 * Copyright © 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "vk_frame_pacer.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/ringbuffer.h"
#include "vulkan/vulkan_core.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_physical_device.h"
#include "vk_queue.h"

#define MAX_FRAMES      8
#define MAX_SUBMISSIONS 256
#define MAX_QUERIES     (MAX_SUBMISSIONS * 2)

struct query {
   uint64_t submit_ts;
   uint64_t gpu_ts;
   VkCommandBuffer cmdbuffer;
};

typedef struct queue_context {
   bool latency_sensitive;
   VkCommandPool cmdPool;
   VkQueryPool queryPool;
   VkSemaphore semaphore;
   uint64_t semaphore_value;
   uint64_t signal_value[MAX_FRAMES];
   uint8_t num_submits[MAX_FRAMES];
   RINGBUFFER_DECLARE(queries, struct query, MAX_QUERIES);
} queue_context;

struct vk_frame_pacer {
   struct vk_device *device;
   bool enabled;

   struct {
      uint64_t base_device_ts;
      uint64_t base_host_ts;
      float timestamp_period;
   } calibration;

   RINGBUFFER_DECLARE(frames, struct vk_frame_time_info, MAX_FRAMES);
   struct vk_frame_time_info *active_frame;
   struct hash_table queues;

   vk_frame_time_cb update;
   void *user_data;

   PFN_vkQueueSubmit2 queue_submit2;
   PFN_vkQueuePresentKHR queue_present;
   queue_context queue_ctx_data[];
};

void
vk_frame_pacer_calibrate(struct vk_frame_pacer *frame_pacer)
{
   uint64_t device_ts, host_ts;
   struct vk_device *device = frame_pacer->device;

   /* We use a short-cut here: Instead of calling GetCalibratedTimestampsKHR,
    * simply get a device and a host timestamp.
    * This is both faster and more accurate since we ensure that the host timestamp
    * is taken after the device timestamp and only once.
    */
   VkResult result = vk_device_get_timestamp(device, VK_TIME_DOMAIN_DEVICE_KHR, &device_ts);
   vk_device_get_timestamp(device, device->calibrate_time_domain, &host_ts);

   if (result == VK_SUCCESS) {
      /* In order to avoid variance, instead of simply using the new device_ts,
       * we take a moving average over the predicted new device_ts and the
       * measured device_ts.
       */
      uint64_t elapsed = host_ts - frame_pacer->calibration.base_host_ts;
      uint64_t predicted_device_ts =
         frame_pacer->calibration.base_device_ts +
         (uint64_t)((double)elapsed / frame_pacer->calibration.timestamp_period);
      int64_t error = device_ts - predicted_device_ts;
      frame_pacer->calibration.base_host_ts = host_ts;
      frame_pacer->calibration.base_device_ts = predicted_device_ts + error / 4;
   }
}

static bool
evaluate_frame(struct vk_frame_pacer *frame_pacer, struct vk_frame_time_info *frame,
               uint64_t timeout)
{
   struct vk_device *device = frame_pacer->device;
   VkDevice _device = vk_device_to_handle(device);
   assert(frame != frame_pacer->active_frame);
   const uint32_t frame_idx = ringbuffer_index(frame_pacer->frames, frame);

   /* Before we commit to completing a frame, all submits on all queues must have completed. */
   hash_table_foreach(&frame_pacer->queues, entry) {
      queue_context *queue_ctx = entry->data;

      /* Wait for the timeline semaphore of the frame to be signaled. */
      struct VkSemaphoreWaitInfo wait_info = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
         .semaphoreCount = 1,
         .pSemaphores = &queue_ctx->semaphore,
         .pValues = &queue_ctx->signal_value[frame_idx],
      };
      VkResult res = device->dispatch_table.WaitSemaphores(_device, &wait_info, timeout);

      if (res != VK_SUCCESS)
         return false;
   }

   uint64_t cpu_start_time = UINT64_MAX;
   uint64_t gpu_start_time = UINT64_MAX;
   uint64_t gpu_end_time = 0;
   int64_t min_delay = INT64_MAX;

   /* For each queue, retrieve timestamp query results. */
   hash_table_foreach(&frame_pacer->queues, entry) {
      queue_context *queue_ctx = entry->data;

      /* As this function must be externally synchronized and is the only place where queries are
       * free'd, we don't need to lock the query ringbuffer here in order to read the first entry.
       */
      struct query *query_begin = ringbuffer_first(queue_ctx->queries);
      uint32_t query_idx = ringbuffer_index(queue_ctx->queries, query_begin);
      int num_timestamps = MIN2(queue_ctx->num_submits[frame_idx] * 2, MAX_QUERIES - query_idx);

      while (num_timestamps > 0) {
         /* Retrieve timestamp results from this queue. */
         VkResult res = device->dispatch_table.GetQueryPoolResults(
            _device, queue_ctx->queryPool, query_idx, num_timestamps,
            sizeof(struct query) * num_timestamps, &query_begin->gpu_ts, sizeof(struct query),
            VK_QUERY_RESULT_64_BIT);
         assert(res == VK_SUCCESS);

         gpu_start_time = MIN2(gpu_start_time, query_begin->gpu_ts);
         cpu_start_time = MIN2(cpu_start_time, query_begin->submit_ts);

         for (unsigned i = 0; i < num_timestamps; i += 2) {
            /* Calibrate GPU begin timestamp. */
            uint64_t begin_gpu_ts =
               frame_pacer->calibration.base_host_ts +
               (uint64_t)((double)(query_begin->gpu_ts - frame_pacer->calibration.base_device_ts) *
                          (double)frame_pacer->calibration.timestamp_period);

            /* This value might be negative due to timestamp inaccuracies. */
            int64_t submission_delay = begin_gpu_ts - query_begin->submit_ts;
            min_delay = MIN2(min_delay, submission_delay);

            /* Even timestamps mark the begin and odd timestamps mark the end of each submission. */
            struct query *query_end = ringbuffer_next(queue_ctx->queries, query_begin);
            query_begin = ringbuffer_next(queue_ctx->queries, query_end);

            /* Take the maximum of all GPU end timestamps. */
            gpu_end_time = MAX2(gpu_end_time, query_end->gpu_ts);
         }

         /* Reset Query Pool. */
         device->dispatch_table.ResetQueryPool(_device, queue_ctx->queryPool, query_idx,
                                               num_timestamps);
         query_idx = ringbuffer_index(queue_ctx->queries, query_begin);

         /* Free queries. */
         ringbuffer_lock(queue_ctx->queries);
         {
            /* Ensure that the total number of queries across all frames is correct. */
            ASSERTED uint32_t submit_count = 0;
            for (unsigned i = 0; i < MAX_FRAMES; i++)
               submit_count += queue_ctx->num_submits[i];
            assert(2 * submit_count == queue_ctx->queries.size);

            ringbuffer_free_n(queue_ctx->queries, num_timestamps);
         }
         ringbuffer_unlock(queue_ctx->queries);

         /* Process the remaining timestamps if any. */
         queue_ctx->num_submits[frame_idx] -= num_timestamps / 2;
         num_timestamps = queue_ctx->num_submits[frame_idx] * 2;
      }
      assert(queue_ctx->num_submits[frame_idx] == 0);
   }

   /* Populate vk_frame_time_info. */
   if (gpu_end_time) {
      frame->frame_time = (double)(gpu_end_time - gpu_start_time) *
                          (double)frame_pacer->calibration.timestamp_period;
      /* Cap the measured frame time at 10 FPS, in order to keep the application
       * usable if anything went wrong.
       */
      frame->frame_time = MIN2(frame->frame_time, ONE_SECOND_IN_NS / 10);
      frame->min_delay = MAX2(0, min_delay);
   }

   return true;
}

void *
vk_frame_pacer_get_user_data(struct vk_frame_pacer *frame_pacer)
{
   return frame_pacer->user_data;
}

void
vk_frame_pacer_evaluate(struct vk_frame_pacer *frame_pacer)
{
   uint64_t timeout;
   struct vk_frame_time_info *frame;

   ringbuffer_lock(frame_pacer->frames);
   {
      /* Don't try to evaluate the currently active frame. */
      if (frame_pacer->frames.size <= 1) {
         ringbuffer_unlock(frame_pacer->frames);
         return;
      }

      /* If we have at least 3 frames, then 2 frames must already have been
       * submitted to the GPU. It is safe to wait for one to complete.
       */
      frame = ringbuffer_first(frame_pacer->frames);
      /* TODO: use a more fine-grained timeout */
      timeout = frame_pacer->frames.size >= 3 ? UINT64_MAX : 0;
   }
   ringbuffer_unlock(frame_pacer->frames);

   while (evaluate_frame(frame_pacer, frame, timeout)) {
      /* Update user data with most recent vk_frame_time_info. */
      frame_pacer->update(frame, vk_frame_pacer_get_user_data(frame_pacer));

      ringbuffer_lock(frame_pacer->frames);
      {
         ringbuffer_free(frame_pacer->frames, frame);

         if (frame_pacer->frames.size >= 2)
            frame = ringbuffer_first(frame_pacer->frames);
         else
            frame = NULL;
      }
      ringbuffer_unlock(frame_pacer->frames);

      /* If we successfully evaluated one frame, only attempt to evaluate the next one,
       * if the completion is clearly expected before the next deadline.
       * Also, don't use a timeout in this case.
       */
      timeout = 0;
      if (!frame)
         break;
   }
}

static struct query *
allocate_query(queue_context *queue_ctx, uint32_t frame_idx)
{
   /* Allow for a single frame to use at most half of the query pool. */
   if (queue_ctx->num_submits[frame_idx] >= MAX_SUBMISSIONS / 2)
      return NULL;

   return ringbuffer_alloc(queue_ctx->queries);
}

static bool
get_timestamp_queries(struct vk_frame_pacer *frame_pacer, struct queue_context *queue_ctx,
                      VkCommandBuffer timestamps[2], VkSemaphoreSubmitInfo *semaphore_info)
{
   uint64_t now;
   vk_device_get_timestamp(frame_pacer->device, frame_pacer->device->calibrate_time_domain, &now);

   uint32_t frame_idx = ringbuffer_index(frame_pacer->frames, frame_pacer->active_frame);
   struct query *query = NULL;

   /* Allocate query to mark the begin of GPU work. */
   query = allocate_query(queue_ctx, frame_idx);

   if (query == NULL)
      return false;

   /* Assign CPU timestamp and commandBuffer for GPU timestamp. */
   query->submit_ts = now;
   timestamps[0] = query->cmdbuffer;

   /* Allocate a second query to mark the end of GPU work. */
   query = allocate_query(queue_ctx, frame_idx);
   assert(query);
   query->submit_ts = now;
   timestamps[1] = query->cmdbuffer;

   /* Increment timeline semaphore count. */
   *semaphore_info = (VkSemaphoreSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = queue_ctx->semaphore,
      .value = ++queue_ctx->semaphore_value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };

   /* Add new submission entry for the current frame */
   queue_ctx->signal_value[frame_idx] = queue_ctx->semaphore_value;
   queue_ctx->num_submits[frame_idx]++;

   return true;
}

static VkResult
frame_pacer_submit(struct vk_frame_pacer *frame_pacer, VkQueue _queue, uint32_t submitCount,
                   const VkSubmitInfo2 *pSubmits, VkFence fence, bool early_submit, uint32_t first,
                   VkCommandBuffer timestamps[2], VkSemaphoreSubmitInfo *semaphore_info)
{
   VkSubmitInfo2 *submits;
   VkCommandBufferSubmitInfo *cmdbuffers_begin;
   VkCommandBufferSubmitInfo *cmdbuffers_end;
   VkSemaphoreSubmitInfo *semaphores;
   VK_MULTIALLOC(ma);

   if (early_submit) {
      vk_multialloc_add(&ma, &submits, VkSubmitInfo2, submitCount + 1);
      vk_multialloc_add(&ma, &cmdbuffers_begin, VkCommandBufferSubmitInfo, 1);
      vk_multialloc_add(&ma, &cmdbuffers_end, VkCommandBufferSubmitInfo,
                        pSubmits[submitCount - 1].commandBufferInfoCount + 1);
      first = 0;
   } else {
      vk_multialloc_add(&ma, &submits, VkSubmitInfo2, submitCount);
      if (first == submitCount - 1) {
         vk_multialloc_add(&ma, &cmdbuffers_begin, VkCommandBufferSubmitInfo,
                           pSubmits[first].commandBufferInfoCount + 2);
      } else {
         vk_multialloc_add(&ma, &cmdbuffers_begin, VkCommandBufferSubmitInfo,
                           pSubmits[first].commandBufferInfoCount + 1);
         vk_multialloc_add(&ma, &cmdbuffers_end, VkCommandBufferSubmitInfo,
                           pSubmits[submitCount - 1].commandBufferInfoCount + 1);
      }
   }
   vk_multialloc_add(&ma, &semaphores, VkSemaphoreSubmitInfo,
                     pSubmits[submitCount - 1].signalSemaphoreInfoCount + 1);

   void *buf =
      vk_multialloc_zalloc(&ma, &frame_pacer->device->alloc, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (!buf)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   if (early_submit) {
      memcpy(submits + 1, pSubmits, sizeof(VkSubmitInfo2) * submitCount);
      submits[0] = (VkSubmitInfo2){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitCount++;
   } else {
      memcpy(submits, pSubmits, sizeof(VkSubmitInfo2) * submitCount);
   }

   VkSubmitInfo2 *submit_info = &submits[first];

   /* Add timestamp commandbuffer before first submission. */
   cmdbuffers_begin[0] = (VkCommandBufferSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = timestamps[0],
   };
   memcpy(&cmdbuffers_begin[1], submit_info->pCommandBufferInfos,
          sizeof(VkCommandBufferSubmitInfo) * submit_info->commandBufferInfoCount);
   submit_info->pCommandBufferInfos = cmdbuffers_begin;
   submit_info->commandBufferInfoCount++;

   /* Add timestamp commandbuffer after last submission. */
   submit_info = &submits[submitCount - 1];
   if (first == submitCount - 1) {
      cmdbuffers_begin[submit_info->commandBufferInfoCount] = (VkCommandBufferSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
         .commandBuffer = timestamps[1],
      };
   } else {
      memcpy(&cmdbuffers_end[0], submit_info->pCommandBufferInfos,
             sizeof(VkCommandBufferSubmitInfo) * submit_info->commandBufferInfoCount);
      cmdbuffers_end[submit_info->commandBufferInfoCount] = (VkCommandBufferSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
         .commandBuffer = timestamps[1],
      };
      submit_info->pCommandBufferInfos = cmdbuffers_end;
   }
   submit_info->commandBufferInfoCount++;

   /* Add timeline semaphore to last submission. */
   memcpy(semaphores, submit_info->pSignalSemaphoreInfos,
          sizeof(VkSemaphoreSubmitInfo) * submit_info->signalSemaphoreInfoCount);
   semaphores[submit_info->signalSemaphoreInfoCount] = *semaphore_info;
   submit_info->pSignalSemaphoreInfos = semaphores;
   submit_info->signalSemaphoreInfoCount++;

   /* Submit with added timestamp query commandbuffers. */
   VkResult res = frame_pacer->queue_submit2(_queue, submitCount, submits, fence);

   vk_free(&frame_pacer->device->alloc, submits);
   return res;
}

VKAPI_ATTR static VkResult VKAPI_CALL
frame_pacer_QueueSubmit2(VkQueue _queue, uint32_t submitCount, const VkSubmitInfo2 *pSubmits,
                         VkFence fence)
{
   struct vk_queue *queue = vk_queue_from_handle(_queue);
   struct vk_device *device = queue->base.device;
   struct vk_frame_pacer *frame_pacer = device->frame_pacer;

   if (!p_atomic_read(&frame_pacer->enabled) || !submitCount)
      return frame_pacer->queue_submit2(_queue, submitCount, pSubmits, fence);

   bool has_wait_before_cmdbuffer = false;
   int first = -1;

   /* Check if any submission contains commandbuffers. */
   for (unsigned i = 0; i < submitCount; i++) {
      if (pSubmits[i].waitSemaphoreInfoCount != 0)
         has_wait_before_cmdbuffer = true;

      if (pSubmits[i].commandBufferInfoCount) {
         first = i;
         break;
      }
   }

   /* Begin critical section. */
   ringbuffer_lock(frame_pacer->frames);

   struct hash_entry *entry = _mesa_hash_table_search(&frame_pacer->queues, queue);
   if (!frame_pacer->active_frame || !entry) {
      ringbuffer_unlock(frame_pacer->frames);
      return frame_pacer->queue_submit2(_queue, submitCount, pSubmits, fence);
   }

   struct queue_context *queue_ctx = entry->data;
   ringbuffer_lock(queue_ctx->queries);

   /* Don't record timestamps for queues that are not deemed sensitive to latency. */
   bool need_query = frame_pacer->active_frame && p_atomic_read(&queue_ctx->latency_sensitive);

   /* For the very first submissions in a frame (until we observe real GPU work happening),
    * we would want to submit a timestamp before anything else, including waits.
    * This allows us to detect a sensitive queue going idle before we can submit work to it.
    * If the queue in question depends on semaphores from other unrelated queues,
    * we may not easily be able to detect that situation without adding a lot more complexity.
    */
   uint32_t frame_idx = ringbuffer_index(frame_pacer->frames, frame_pacer->active_frame);
   bool early_submit = has_wait_before_cmdbuffer && queue_ctx->num_submits[frame_idx] == 0;
   need_query &= (first >= 0 || early_submit);

   /* Get timestamp commandbuffers. */
   VkCommandBuffer timestamps[2];
   VkSemaphoreSubmitInfo semaphore_info;
   if (!need_query || !get_timestamp_queries(frame_pacer, queue_ctx, timestamps, &semaphore_info)) {
      ringbuffer_unlock(queue_ctx->queries);
      ringbuffer_unlock(frame_pacer->frames);
      return frame_pacer->queue_submit2(_queue, submitCount, pSubmits, fence);
   }

   /* End critical section for externally synchronized queues:
    * As we incremented the timeline semaphore value, we know for sure that the frame_pacer
    * won't be disabled until this submission completed and got signalled. Thus, it's safe
    * to use the semaphore without synchronization.
    */
   if (!(queue->flags & VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR))
      ringbuffer_unlock(queue_ctx->queries);
   ringbuffer_unlock(frame_pacer->frames);

   /* Submit with added timestamp query commandbuffers. */
   VkResult res = frame_pacer_submit(frame_pacer, _queue, submitCount, pSubmits, fence,
                                     early_submit, first, timestamps, &semaphore_info);

   /* For internally synchronized queues, make sure that the submissions are kept
    * in-order, so that the semaphore can't get destroyed before the submission
    * completed.
    */
   if (queue->flags & VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR)
      ringbuffer_unlock(queue_ctx->queries);

   return res;
}

VKAPI_ATTR static VkResult VKAPI_CALL
frame_pacer_QueuePresentKHR(VkQueue _queue, const VkPresentInfoKHR *pPresentInfo)
{
   struct vk_queue *queue = vk_queue_from_handle(_queue);
   struct vk_device *device = queue->base.device;
   struct vk_frame_pacer *frame_pacer = device->frame_pacer;

   if (p_atomic_read(&frame_pacer->enabled)) {
      ringbuffer_lock(frame_pacer->frames);
      {
         /* When multiple queues are in flight, the min-delay approach
          * has problems. An async compute queue could be submitted to
          * with very low delay while the main graphics queue would be swamped with work.
          * If we take a global min-delay over all queues, the algorithm would
          * assume that there is very low delay and thus sleeps are disabled, but
          * unless the graphics work depends directly on the async compute work,
          * this is a false assumption.
          */
         struct hash_entry *entry = _mesa_hash_table_search(&frame_pacer->queues, queue);
         if (entry)
            p_atomic_set(&((queue_context *)entry->data)->latency_sensitive, true);

         /* Initialize new frame. */
         if (p_atomic_read(&frame_pacer->enabled)) {
            struct vk_frame_time_info *frame = ringbuffer_alloc(frame_pacer->frames);
            if (frame) {
               frame->frame_time = 0;
               frame->min_delay = 0;
            }
            frame_pacer->active_frame = frame;
         }
      }
      ringbuffer_unlock(frame_pacer->frames);
   }

   return frame_pacer->queue_present(_queue, pPresentInfo);
}

void
vk_frame_pacer_disable(struct vk_frame_pacer *frame_pacer)
{
   ringbuffer_lock(frame_pacer->frames);
   {
      p_atomic_set(&frame_pacer->enabled, false);
      frame_pacer->active_frame = NULL;

      /* Evaluate all pending frames.  This ensures that all CommandBuffers have completed,
       * all Semaphores have been signaled and the data from the QueryPools have been read.
       */
      while (frame_pacer->frames.size) {
         /* Set timeout=UINT64_MAX, so that all pending timestamp queries get completed. */
         struct vk_frame_time_info *frame = ringbuffer_first(frame_pacer->frames);
         evaluate_frame(frame_pacer, frame, UINT64_MAX);
         ringbuffer_free(frame_pacer->frames, frame);
      }
   }
   ringbuffer_unlock(frame_pacer->frames);
}

/* Initialize per-queue context:
 *
 * This includes creating one CommandPool and one QueryPool per Queue as well as
 * recording one CommandBuffer per timestamp query.
 */
static VkResult
init_queue_context(struct vk_device *device, queue_context *queue_ctx, unsigned family_idx)
{
#define CHECK_RESULT(res, label)                                                                   \
   if (res != VK_SUCCESS) {                                                                        \
      goto label;                                                                                  \
   }

   VkResult result;
   VkDevice _device = vk_device_to_handle(device);

   /* Create command pool */
   struct VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .queueFamilyIndex = family_idx,
   };
   result = device->dispatch_table.CreateCommandPool(_device, &pool_info, &device->alloc,
                                                     &queue_ctx->cmdPool);
   CHECK_RESULT(result, fail_cmdpool)

   /* Create query pool */
   VkQueryPoolCreateInfo query_pool_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = MAX_QUERIES,
   };
   result = device->dispatch_table.CreateQueryPool(_device, &query_pool_info, &device->alloc,
                                                   &queue_ctx->queryPool);
   CHECK_RESULT(result, fail_querypool)
   device->dispatch_table.ResetQueryPool(_device, queue_ctx->queryPool, 0, MAX_QUERIES);
   ringbuffer_init(queue_ctx->queries);

   /* Create timeline semaphore */
   VkSemaphoreTypeCreateInfo timelineCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = NULL,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
   };
   VkSemaphoreCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timelineCreateInfo,
      .flags = 0,
   };
   result = device->dispatch_table.CreateSemaphore(_device, &createInfo, &device->alloc,
                                                   &queue_ctx->semaphore);
   CHECK_RESULT(result, fail_semaphore);

   for (unsigned j = 0; j < MAX_QUERIES; j++) {
      struct query *query = &queue_ctx->queries.data[j];

      /* Allocate commandBuffer for timestamp. */
      VkCommandBufferAllocateInfo buffer_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = queue_ctx->cmdPool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      result =
         device->dispatch_table.AllocateCommandBuffers(_device, &buffer_info, &query->cmdbuffer);
      CHECK_RESULT(result, fail)

      /* Record commandbuffer. */
      VkCommandBufferBeginInfo beginInfo = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      };

      result = device->dispatch_table.BeginCommandBuffer(query->cmdbuffer, &beginInfo);
      CHECK_RESULT(result, fail)
      if (j % 2 == 0)
         device->dispatch_table.CmdWriteTimestamp(
            query->cmdbuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queue_ctx->queryPool, j);
      else
         device->dispatch_table.CmdWriteTimestamp(
            query->cmdbuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queue_ctx->queryPool, j);
      result = device->dispatch_table.EndCommandBuffer(query->cmdbuffer);
      CHECK_RESULT(result, fail)
   }

#undef CHECK_RESULT
   return result;

fail:
   device->dispatch_table.DestroySemaphore(_device, queue_ctx->semaphore, &device->alloc);
fail_semaphore:
   device->dispatch_table.DestroyQueryPool(_device, queue_ctx->queryPool, &device->alloc);
fail_querypool:
   device->dispatch_table.DestroyCommandPool(_device, queue_ctx->cmdPool, &device->alloc);
fail_cmdpool:
   return result;
}

static void
finish_queue_context(struct hash_entry *entry)
{
   struct vk_queue *queue = (struct vk_queue *)entry->key;
   struct vk_device *device = queue->base.device;
   queue_context *ctx = entry->data;
   VkDevice _device = vk_device_to_handle(device);
   device->dispatch_table.DestroyQueryPool(_device, ctx->queryPool, &device->alloc);
   device->dispatch_table.DestroyCommandPool(_device, ctx->cmdPool, &device->alloc);
   device->dispatch_table.DestroySemaphore(_device, ctx->semaphore, &device->alloc);
}

void
vk_frame_pacer_enable(struct vk_frame_pacer *frame_pacer)
{
   if (p_atomic_read(&frame_pacer->enabled))
      return;

   if (frame_pacer->queues.entries) {
      p_atomic_set(&frame_pacer->enabled, true);
      return;
   }

   /* Check Queue properties. */
   struct vk_device *device = frame_pacer->device;
   VkPhysicalDevice pdevice = vk_physical_device_to_handle(device->physical);
   unsigned max_queue_family_count = 64;
   VkQueueFamilyProperties queue_properties[64];
   device->physical->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
      pdevice, &max_queue_family_count, queue_properties);

   /* Initialize Queue contexts. */
   VkResult result;
   unsigned idx = 0;
   list_for_each_entry(struct vk_queue, queue, &device->queues, link) {
      /* Only consider graphics and compute queues. */
      unsigned index = queue->queue_family_index;
      if (index >= max_queue_family_count)
         continue;
      if (!(queue_properties[index].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
         continue;

      assert(queue_properties[index].timestampValidBits == 64);
      result = init_queue_context(device, &frame_pacer->queue_ctx_data[idx], index);
      if (result != VK_SUCCESS) {
         _mesa_hash_table_clear(&frame_pacer->queues, finish_queue_context);
         return;
      }
      _mesa_hash_table_insert(&frame_pacer->queues, queue, &frame_pacer->queue_ctx_data[idx++]);
   }

   /* Get calibrated timestamps. */
   result = vk_device_get_timestamp(device, VK_TIME_DOMAIN_DEVICE_KHR,
                                    &frame_pacer->calibration.base_device_ts);
   vk_device_get_timestamp(device, device->calibrate_time_domain,
                           &frame_pacer->calibration.base_host_ts);

   p_atomic_set(&frame_pacer->enabled, result == VK_SUCCESS);
}

void
vk_frame_pacer_finish(struct vk_device *device)
{
   struct vk_frame_pacer *frame_pacer = device->frame_pacer;
   vk_frame_pacer_disable(frame_pacer);

   /* Clean up query pools, command pools and semaphores. */
   _mesa_hash_table_fini(&frame_pacer->queues, finish_queue_context);

   vk_free(&device->alloc, frame_pacer);
   device->frame_pacer = NULL;
}

VkResult
vk_frame_pacer_init(struct vk_device *device, vk_frame_time_cb update_cb, size_t data_size)
{
   assert(update_cb);

   /* Allocate the context. */
   struct vk_frame_pacer *frame_pacer;
   queue_context *queues;
   uint8_t *user_data;
   unsigned num_queues = list_length(&device->queues);
   VK_MULTIALLOC(ma);
   vk_multialloc_add(&ma, &frame_pacer, struct vk_frame_pacer, 1);
   vk_multialloc_add(&ma, &queues, queue_context, num_queues);
   vk_multialloc_add_size(&ma, &user_data, uint8_t, data_size);
   void *buf = vk_multialloc_zalloc(&ma, &device->alloc, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!buf)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   device->frame_pacer = frame_pacer;
   frame_pacer->device = device;
   frame_pacer->calibration.timestamp_period = device->physical->properties.timestampPeriod;
   ringbuffer_init(frame_pacer->frames);
   _mesa_pointer_hash_table_init(&frame_pacer->queues, NULL);
   frame_pacer->update = update_cb;
   frame_pacer->user_data = user_data;

   /* Update dispatch table. */
   frame_pacer->queue_submit2 = device->dispatch_table.QueueSubmit2;
   frame_pacer->queue_present = device->dispatch_table.QueuePresentKHR;
   device->dispatch_table.QueueSubmit2 = frame_pacer_QueueSubmit2;
   device->dispatch_table.QueuePresentKHR = frame_pacer_QueuePresentKHR;

   return VK_SUCCESS;
}