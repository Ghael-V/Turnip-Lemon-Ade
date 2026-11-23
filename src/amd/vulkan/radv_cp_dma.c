/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based on si_state.c
 * Copyright © 2015 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_cp_dma.h"
#include "tools/radv_debug_hang.h"
#include "tools/radv_sqtt.h"
#include "radv_buffer.h"
#include "radv_cs.h"
#include "radv_shader.h"
#include "radv_tracepoints.h"
#include "sid.h"

/* This will cause CP to wait until CP DMA is done. */
#define CP_DMA_SYNC   (1 << 0)
#define CP_DMA_USE_L2 (1 << 1)
#define CP_DMA_CLEAR  (1 << 2)

#define CP_DMA_SAVE_HEADER_PTR_FOR_SYNC (1 << 4)

/* Alignment for optimal performance. */
#define SI_CPDMA_ALIGNMENT 32

/* The max number of bytes that can be copied per packet. */
static inline unsigned
cp_dma_max_byte_count(const struct radv_physical_device *pdev)
{
   unsigned max = pdev->info.gfx_level >= GFX11 && !pdev->drirc.debug.gfx11_full_size_cp_dma ? 32767
                  : pdev->info.gfx_level >= GFX9                                             ? S_506_BYTE_COUNT(~0u)
                                                                                             : S_415_BYTE_COUNT(~0u);

   /* make it aligned for optimal performance */
   return max & ~(SI_CPDMA_ALIGNMENT - 1);
}

/* Emit a CP DMA packet to do a copy from one buffer to another, or to clear
 * a buffer. The size must fit in bits [20:0]. If CP_DMA_CLEAR is set, src_va is a 32-bit
 * clear value.
 */
static void
radv_emit_cp_dma(struct radv_cmd_buffer *cmd_buffer, uint64_t dst_va, uint64_t src_va, unsigned size, unsigned flags)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   const bool cp_dma_use_L2 = (flags & CP_DMA_USE_L2) && pdev->info.cp_dma_use_L2;
   const bool cp_dma_use_mall = pdev->info.gfx_level == GFX12;
   /* GFX12: TC_L2 means MALL, which should always be set. */
   const bool cp_dma_tc_l2_flag = cp_dma_use_L2 || cp_dma_use_mall;
   uint32_t header = 0, command = 0;

   assert(size <= cp_dma_max_byte_count(pdev));

   radeon_check_space(device->ws, cs->b, 9);
   if (pdev->info.gfx_level >= GFX9)
      command |= S_506_BYTE_COUNT(size);
   else
      command |= S_415_BYTE_COUNT(size);

   /* Sync flags. Only present for PFP/ME. MEC always syncs. */
   if (!cmd_buffer->is_mec && flags & CP_DMA_SYNC)
      header |= S_501_CP_SYNC(1);

   /* Src and dst flags. */
   if (cp_dma_tc_l2_flag)
      header |= S_501_DST_SEL(V_501_DST_ADDR_USING_L2);

   if (flags & CP_DMA_CLEAR)
      header |= S_501_SRC_SEL(V_501_DATA);
   else if (cp_dma_tc_l2_flag)
      header |= S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2);

   radeon_begin(cs);
   if (pdev->info.gfx_level >= GFX7) {
      radeon_emit(PKT3(PKT3_DMA_DATA, 5, 0));
      if (!cmd_buffer->is_mec && flags & CP_DMA_SAVE_HEADER_PTR_FOR_SYNC)
         device->ws->cs_set_last_cp_dma_header(cs->b, &__cs_buf[__cs_num]);
      radeon_emit(header);
      radeon_emit(src_va);       /* SRC_ADDR_LO [31:0] */
      radeon_emit(src_va >> 32); /* SRC_ADDR_HI [31:0] */
      radeon_emit(dst_va);       /* DST_ADDR_LO [31:0] */
      radeon_emit(dst_va >> 32); /* DST_ADDR_HI [31:0] */
      radeon_emit(command);
   } else {
      assert(!cp_dma_tc_l2_flag);
      header |= S_412_SRC_ADDR_HI(src_va >> 32);
      radeon_emit(PKT3(PKT3_CP_DMA, 4, 0));
      radeon_emit(src_va); /* SRC_ADDR_LO [31:0] */
      if (!cmd_buffer->is_mec && flags & CP_DMA_SAVE_HEADER_PTR_FOR_SYNC)
         device->ws->cs_set_last_cp_dma_header(cs->b, &__cs_buf[__cs_num]);
      radeon_emit(header);                  /* SRC_ADDR_HI [15:0] + flags. */
      radeon_emit(dst_va);                  /* DST_ADDR_LO [31:0] */
      radeon_emit((dst_va >> 32) & 0xffff); /* DST_ADDR_HI [15:0] */
      radeon_emit(command);
   }
   radeon_end();

   if (radv_device_fault_detection_enabled(device))
      radv_cmd_buffer_trace_emit(cmd_buffer);
}

/* Emit a CP DMA prefetch.
 * This is useful for warming up caches before draw commands,
 * for example we use it to load shader binaries and VBO descriptors into the cache.
 * Implemented by starting a CP DMA copy where the source and destination are the same.
 *
 * On GPUs where CP DMA uses L2, it loads binaries into L2.
 * On GPUs that have MALL (infinity cache), it loads binaries into MALL.
 *
 * More specifically:
 *
 * | GPU generation  | CP DMA L2 | MALL | Prefetch location |
 * | --------------- | --------- | ---- | ----------------- |
 * | GFX6            | -         | -    | -                 |
 * | GFX7 - 10       | yes       | -    | L2                |
 * | GFX10.3 - 11.5  | yes       | yes  | L2, MALL          |
 * | GFX12           | -         | yes  | MALL              |
 *
 */
void
radv_cs_cp_dma_prefetch(const struct radv_device *device, struct radv_cmd_stream *cs, uint64_t va, unsigned size)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radeon_winsys *ws = device->ws;
   enum amd_gfx_level gfx_level = pdev->info.gfx_level;
   uint32_t header = 0, command = 0;

   if (gfx_level >= GFX11)
      size = MIN2(size, 32768 - SI_CPDMA_ALIGNMENT);

   assert(size <= cp_dma_max_byte_count(pdev));

   radeon_check_space(ws, cs->b, 9);

   uint64_t aligned_va = va & ~(SI_CPDMA_ALIGNMENT - 1);
   uint64_t aligned_size = ((va + size + SI_CPDMA_ALIGNMENT - 1) & ~(SI_CPDMA_ALIGNMENT - 1)) - aligned_va;

   if (gfx_level >= GFX9) {
      command |= S_506_BYTE_COUNT(aligned_size) | S_506_DISABLE_WR_CONFIRM(1);
      header |= S_501_DST_SEL(V_501_DST_NOWHERE);
   } else {
      command |= S_415_BYTE_COUNT(aligned_size) | S_415_DISABLE_WR_CONFIRM(1);
      header |= S_501_DST_SEL(V_501_DST_ADDR_USING_L2);
   }

   header |= S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2);

   radeon_begin(cs);
   radeon_emit(PKT3(PKT3_DMA_DATA, 5, 0));
   radeon_emit(header);
   radeon_emit(aligned_va);       /* SRC_ADDR_LO [31:0] */
   radeon_emit(aligned_va >> 32); /* SRC_ADDR_HI [31:0] */
   radeon_emit(aligned_va);       /* DST_ADDR_LO [31:0] */
   radeon_emit(aligned_va >> 32); /* DST_ADDR_HI [31:0] */
   radeon_emit(command);
   radeon_end();
}

static void
radv_cp_dma_realign_engine(struct radv_cmd_buffer *cmd_buffer, unsigned size)
{
   uint64_t va;
   unsigned buf_size = SI_CPDMA_ALIGNMENT * 2;

   assert(size < SI_CPDMA_ALIGNMENT);

   radv_cmd_buffer_vram_suballoc(cmd_buffer, buf_size, SI_CPDMA_ALIGNMENT, &va);
   radv_emit_cp_dma(cmd_buffer, va, va + SI_CPDMA_ALIGNMENT, size, 0);
}

void
radv_cp_dma_copy_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dest_va, uint64_t size)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint64_t main_src_va, main_dest_va;
   uint64_t skipped_size = 0, realign_size = 0;

   if (!pdev->info.cp_dma_use_L2) {
      /* Invalidate L2 in case "src_va" or "dest_va" were previously written through L2. */
      cmd_buffer->state.flush_bits |= AC_BARRIER_INV_L2;
   }

   if (pdev->info.has_cp_dma_unaligned_copy_perf_issue) {
      /* If the size is not aligned, we must add a dummy copy at the end
       * just to align the internal counter. Otherwise, the DMA engine
       * would slow down by an order of magnitude for following copies.
       */
      if (size % SI_CPDMA_ALIGNMENT)
         realign_size = SI_CPDMA_ALIGNMENT - (size % SI_CPDMA_ALIGNMENT);

      /* If the copy begins unaligned, we must start copying from the next
       * aligned block and the skipped part should be copied after everything
       * else has been copied. Only the src alignment matters, not dst.
       */
      if (src_va % SI_CPDMA_ALIGNMENT) {
         skipped_size = SI_CPDMA_ALIGNMENT - (src_va % SI_CPDMA_ALIGNMENT);
         /* The main part will be skipped if the size is too small. */
         skipped_size = MIN2(skipped_size, size);
         size -= skipped_size;
      }
   }
   main_src_va = src_va + skipped_size;
   main_dest_va = dest_va + skipped_size;

   radv_utrace_begin_cp_dma_copy_memory(cmd_buffer, size);

   /* Flush the caches for the first copy only. */
   if (cmd_buffer->state.flush_bits)
      radv_emit_cache_flush(cmd_buffer, false);

   while (size) {
      unsigned byte_count = MIN2(size, cp_dma_max_byte_count(pdev));

      radv_emit_cp_dma(cmd_buffer, main_dest_va, main_src_va, byte_count,
                       CP_DMA_USE_L2 | (!skipped_size ? CP_DMA_SAVE_HEADER_PTR_FOR_SYNC : 0));

      size -= byte_count;
      main_src_va += byte_count;
      main_dest_va += byte_count;
   }

   if (skipped_size)
      radv_emit_cp_dma(cmd_buffer, dest_va, src_va, skipped_size, CP_DMA_USE_L2 | CP_DMA_SAVE_HEADER_PTR_FOR_SYNC);

   if (realign_size) {
      radv_cp_dma_realign_engine(cmd_buffer, realign_size);
      cmd_buffer->state.cp_dma_realignment_is_busy = !cmd_buffer->is_mec;
   }

   radv_utrace_end_cp_dma_copy_memory(cmd_buffer);
   cmd_buffer->state.dma_is_busy = !cmd_buffer->is_mec;
}

void
radv_cp_dma_fill_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, unsigned value,
                        bool cb_db_cp_coherent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   if (!size)
      return;

   const bool use_L2 = pdev->info.gfx_level >= GFX9 || !cb_db_cp_coherent;

   if (use_L2 && !pdev->info.cp_dma_use_L2) {
      /* Invalidate L2 in case "va" was previously written through L2. */
      cmd_buffer->state.flush_bits |= AC_BARRIER_INV_L2;
   }

   assert(va % 4 == 0 && size % 4 == 0);

   /* Flush the caches for the first copy only. */
   if (cmd_buffer->state.flush_bits)
      radv_emit_cache_flush(cmd_buffer, false);

   while (size) {
      unsigned byte_count = MIN2(size, cp_dma_max_byte_count(pdev));

      radv_emit_cp_dma(cmd_buffer, va, value, byte_count,
                       CP_DMA_CLEAR | (use_L2 ? CP_DMA_USE_L2 : 0) | CP_DMA_SAVE_HEADER_PTR_FOR_SYNC);

      size -= byte_count;
      va += byte_count;
   }

   cmd_buffer->state.dma_is_busy = !cmd_buffer->is_mec;
}

void
radv_cp_dma_wait_for_idle(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   if (!cmd_buffer->state.dma_is_busy)
      return;

   assert(!cmd_buffer->is_mec);

   uint32_t *last_cp_dma_header = device->ws->cs_get_last_cp_dma_header(cmd_buffer->cs->b);

   if (last_cp_dma_header && cmd_buffer->cs->b->buf < last_cp_dma_header &&
       cmd_buffer->cs->b->buf + cmd_buffer->cs->b->cdw > last_cp_dma_header &&
       cmd_buffer->cs->b->buf + cmd_buffer->cs->b->cdw - last_cp_dma_header < 30) {
      /* Sync in the last CP DMA packet because it's close. */
      *last_cp_dma_header |= S_501_CP_SYNC(1);
   } else {
      /* Issue a dummy DMA that copies zero bytes.
       *
       * The DMA engine will see that there's no work to do and skip this
       * DMA request, however, the CP will see the sync flag and still wait
       * for all DMAs to complete.
       */
      radv_emit_cp_dma(cmd_buffer, 0, 0, 0, CP_DMA_SYNC);

      /* Only the explicit sync packet guarantees that the realignment is done. */
      cmd_buffer->state.cp_dma_realignment_is_busy = false;
   }

   cmd_buffer->state.dma_is_busy = false;
   device->ws->cs_set_last_cp_dma_header(cmd_buffer->cs->b, NULL);
}
