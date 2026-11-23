/*
 * Copyright © 2020 Valve Corporation
 * SPDX-License-Identifier: MIT
 */


#include "util/log.h"

#include "crashdec.h"


static void
dump_mem_pool_reg_write(unsigned reg, uint32_t data, unsigned context,
                        bool pipe)
{
   /* TODO deal better somehow w/ 64b regs: */
   struct regacc r = {
         .rnn = pipe ? rnn_pipe : NULL,
         .regbase = reg,
         .value = data,
   };
   if (pipe) {
      struct rnndecaddrinfo *info = rnn_reginfo(rnn_pipe, reg);
      printf("\t\twrite %s (%02x) pipe\n", info->name, reg);

      if (!info->typeinfo || strcmp(info->typeinfo->name, "void")) {
         printf("\t\t\t");
         dump_register(&r);
      } else {
         /* registers that ignore their payload */
      }
      rnn_reginfo_free(info);
   } else {
      printf("\t\twrite %s (%05x)", regname(reg, 1), reg);

      if (is_a6xx()) {
         printf(" context %d", context);
      }

      printf("\n");
      dump_register_val(&r, 2);
   }
}

static void
dump_mem_pool_chunk(const uint32_t *chunk)
{
   struct __attribute__((packed)) {
      bool reg0_enabled : 1;
      bool reg1_enabled : 1;
      uint32_t data0 : 32;
      uint32_t data1 : 32;
      uint32_t reg0 : 18;
      uint32_t reg1 : 18;
      bool reg0_pipe : 1;
      bool reg1_pipe : 1;
      uint32_t reg0_context : 1;
      uint32_t reg1_context : 1;
      uint32_t padding : 22;
   } fields;

   memcpy(&fields, chunk, 4 * sizeof(uint32_t));

   if (fields.reg0_enabled) {
      dump_mem_pool_reg_write(fields.reg0, fields.data0, fields.reg0_context,
                              fields.reg0_pipe);
   }

   if (fields.reg1_enabled) {
      dump_mem_pool_reg_write(fields.reg1, fields.data1, fields.reg1_context,
                              fields.reg1_pipe);
   }
}

/* The mem pool comes in a few different sizes, and the size determines the
 * offsets of the various sub-structures within the dump.  The chunk data
 * always starts at offset 0 and is followed by the next-pointers array.
 */
enum mempool_size {
   MEMPOOL_FULL,
   MEMPOOL_HALF,
   MEMPOOL_QUARTER,
};

struct mempool_layout {
   const char *name;
   uint32_t next_pointers_offset;
   uint32_t data1_offset;
   uint32_t data2_offset;
   /* Maximum number of blocks in the pool, also the size of the pointers
    * array.
    */
   uint32_t num_blocks;
   /* log2 of number of chunks per block. */
   uint32_t block_size_log2;
};

static const struct mempool_layout mempool_layouts[] = {
   [MEMPOOL_FULL]    = { "full",    0x1000, 0x1800, 0x2000, 0x80, 3 },
   [MEMPOOL_HALF]    = { "half",    0x800,  0xc00,  0x1000, 0x30, 3 },
   [MEMPOOL_QUARTER] = { "quarter", 0x400,  0x600,  0x800,  0x40, 2 },
};

/* Unfortunately the per-queue state is a little more complicated than
 * a simple pair of begin/end pointers. Instead of a single beginning
 * block, there are *two*, with the property that either the two are
 * equal or the second is the "next" of the first. Similarly there are
 * two end blocks. Thus the queue either looks like this:
 *
 * A -> B -> ... -> C -> D
 *
 * Or like this, or some combination:
 *
 * A/B -> ... -> C/D
 *
 * However, there's only one beginning/end chunk offset. Now the
 * question is, which of A or B is the actual start? I.e. is the chunk
 * offset an offset inside A or B? It depends. I'll show a typical read
 * cycle, starting here (read pointer marked with a *) with a chunk
 * offset of 0:
 *
 *	  A                    B
 *  _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _
 * |_|_|_|_|_|_|_|_| -> |*|_|_|_|_|_|_|_| -> |_|_|_|_|_|_|_|_|
 *
 * Once the pointer advances far enough, the hardware decides to free
 * A, after which the read-side state looks like:
 *
 *	(free)                A/B
 *  _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _
 * |_|_|_|_|_|_|_|_|    |_|_|_|*|_|_|_|_| -> |_|_|_|_|_|_|_|_|
 *
 * Then after advancing the pointer a bit more, the hardware fetches
 * the "next" pointer for A and stores it in B:
 *
 *	(free)                 A                     B
 *  _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _
 * |_|_|_|_|_|_|_|_|    |_|_|_|_|_|_|_|*| -> |_|_|_|_|_|_|_|_|
 *
 * Then the read pointer advances into B, at which point we've come
 * back to the first state having advanced a whole block:
 *
 *	(free)                 A                     B
 *  _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _
 * |_|_|_|_|_|_|_|_|    |_|_|_|_|_|_|_|_| -> |*|_|_|_|_|_|_|_|
 *
 *
 * There is a similar cycle for the write pointer. Now, the question
 * is, how do we know which state we're in? We need to know this to
 * know whether the pointer (*) is in A or B if they're different. It
 * seems like there should be some bit somewhere describing this, but
 * after lots of experimentation I've come up empty-handed. For now we
 * assume that if the pointer is in the first half, then we're in
 * either the first or second state and use B, and otherwise we're in
 * the second or third state and use A. So far I haven't seen anything
 * that violates this assumption.
 */
struct mempool_data1 {
   uint32_t unk0;
   uint32_t padding0[7]; /* Mirrors of unk0 */

   uint32_t writer_first_block_chunk[7];
   uint32_t padding1[1]; /* Mirror of writer_first_block_chunk[6] */

   uint32_t unk1;
   uint32_t padding2[7]; /* Mirrors of unk1 */

   uint32_t writer_second_block[7];
   uint32_t padding3[1];

   uint32_t unk2[7];
   uint32_t padding4[1];

   uint32_t reader_first_block_chunk[7];
   uint32_t padding5[1]; /* Mirror of reader_first_block_chunk[6] */

   uint32_t unk3;
   uint32_t padding6[7]; /* Mirrors of unk3 */

   uint32_t reader_second_block[7];
   uint32_t padding7[1];

   uint32_t block_count[7];
   uint32_t padding[1];

   uint32_t unk4;
   uint32_t padding9[7]; /* Mirrors of unk4 */
};

struct mempool_data2 {
   /* Based on the kernel, the first dword is the mem pool size (in
    * blocks?) and mirrors CP_MEM_POOL_DBG_SIZE.
    */
   uint32_t size;
   uint32_t padding0[7]; /* Mirrors of size */

   uint32_t unk0[2][8];

   /* This seems to be the size of each queue in chunks.  Only the first
    * num_queues entries are read.
    */
   uint32_t queue_sizes[8];

   uint32_t unk1[8][8];
};

/**
 * Walk one queue, following the linked list of blocks from the reader
 * position to the writer position.  Returns the number of chunks visited,
 * or -1 if the walk hit an out of range block index or failed to terminate
 * (which indicates that we've mis-detected the pool layout, or that the
 * dump is corrupt).
 *
 * With dump=false nothing is printed, so that this can also be used to
 * validate a candidate layout.
 */
static int
walk_queue(const uint32_t *mempool, const struct mempool_layout *l,
           const struct mempool_data1 *data1,
           const struct mempool_data2 *data2, unsigned queue, bool dump)
{
   const uint32_t *next_pointers = &mempool[l->next_pointers_offset];
   const uint32_t block_size = 1 << l->block_size_log2;

   uint32_t writer_first_block =
      data1->writer_first_block_chunk[queue] >> l->block_size_log2;
   uint32_t writer_chunk =
      data1->writer_first_block_chunk[queue] & (block_size - 1);
   uint32_t reader_first_block =
      data1->reader_first_block_chunk[queue] >> l->block_size_log2;
   uint32_t reader_chunk =
      data1->reader_first_block_chunk[queue] & (block_size - 1);

   if (dump && verbose) {
      printf("\t\twriter_first_block: 0x%x\n", writer_first_block);
      printf("\t\twriter_second_block: 0x%x\n",
             data1->writer_second_block[queue]);
      printf("\t\twriter_chunk: %d\n", writer_chunk);
      printf("\t\treader_first_block: 0x%x\n", reader_first_block);
      printf("\t\treader_second_block: 0x%x\n",
             data1->reader_second_block[queue]);
      printf("\t\treader_chunk: %d\n", reader_chunk);
      printf("\t\tblock_count: %d\n", data1->block_count[queue]);
      printf("\t\tunk2: 0x%x\n", data1->unk2[queue]);
      printf("\t\tqueue_size: %d\n\n", data2->queue_sizes[queue]);
   }

   uint32_t cur_chunk = reader_chunk;
   uint32_t cur_block = cur_chunk >= block_size / 2 ?
      reader_first_block : data1->reader_second_block[queue];
   uint32_t last_chunk = writer_chunk;
   uint32_t last_block = last_chunk >= block_size / 2 ?
      writer_first_block : data1->writer_second_block[queue];

   if (dump && verbose)
      printf("\tblock %x\n", cur_block);

   if ((cur_block >= l->num_blocks) || (last_block >= l->num_blocks))
      return -1;

   const uint32_t max_chunks = l->num_blocks << l->block_size_log2;
   uint32_t count = 0;

   while ((cur_block != last_block) || (cur_chunk != last_chunk)) {
      if (count++ >= max_chunks)
         return -1;

      const uint32_t *chunk_ptr =
         &mempool[(cur_block * block_size + cur_chunk) * 4];

      if (dump) {
         dump_mem_pool_chunk(chunk_ptr);

         printf("\t%05x: %08x %08x %08x %08x\n",
                16 * (cur_block * block_size + cur_chunk), chunk_ptr[0],
                chunk_ptr[1], chunk_ptr[2], chunk_ptr[3]);
      }

      cur_chunk++;
      if (cur_chunk == block_size) {
         cur_block = next_pointers[cur_block];
         if (dump && verbose)
            printf("\tblock %x\n", cur_block);
         if (cur_block >= l->num_blocks)
            return -1;
         cur_chunk = 0;
      }
   }

   return count;
}

/**
 * Check whether the dump is big enough to contain all of the sub-structures
 * that the given layout expects.
 */
static bool
layout_fits(uint32_t sizedwords, const struct mempool_layout *l)
{
   const size_t data1_dwords = sizeof(struct mempool_data1) / 4;
   const size_t data2_dwords = sizeof(struct mempool_data2) / 4;

   return ((l->data2_offset + data2_dwords) <= sizedwords) &&
          ((l->data1_offset + data1_dwords) <= sizedwords) &&
          ((l->next_pointers_offset + l->num_blocks) <= sizedwords);
}

/**
 * Returns the number of queues whose walked length matches the length
 * recorded in the dump, or -1 if the layout isn't even self-consistent.
 * Used to pick between the possible pool layouts.
 */
static int
validate_layout(const uint32_t *mempool, uint32_t sizedwords,
                const struct mempool_layout *l, unsigned num_queues)
{
   if (!layout_fits(sizedwords, l))
      return -1;

   struct mempool_data1 data1;
   memcpy(&data1, &mempool[l->data1_offset], sizeof(data1));

   struct mempool_data2 data2;
   memcpy(&data2, &mempool[l->data2_offset], sizeof(data2));

   int matches = 0;
   for (unsigned queue = 0; queue < num_queues; queue++) {
      int count = walk_queue(mempool, l, &data1, &data2, queue, false);
      if (count < 0)
         return -1;
      if ((uint32_t)count == data2.queue_sizes[queue])
         matches++;
   }

   return matches;
}

/**
 * Figure out which mem pool layout this dump uses.
 *
 * Some parts have a smaller mem pool, which shifts the offsets of everything
 * after the chunk data.  Guessing wrong doesn't just mis-print a few values:
 * the next-pointers array and the per-queue state end up being read out of
 * unrelated (often mirrored/padding) parts of the dump, so the queue walk
 * follows a bogus linked list.
 *
 * Fortunately we don't have to guess.  The per-queue sizes recorded in data2
 * act as a checksum: walk each queue and see whether the lengths agree.
 */
static enum mempool_size
detect_mempool_size(const uint32_t *mempool, uint32_t sizedwords, bool is_bv,
                    unsigned num_queues)
{
   const enum mempool_size dflt = is_bv ? MEMPOOL_QUARTER : MEMPOOL_FULL;

   unsigned order[ARRAY_SIZE(mempool_layouts) + 2];
   unsigned n = 0;

   /* The kernel stuffs the value of CP_MEM_POOL_SIZE, ie. the number of
    * blocks in the pool, into dword 0x2000 of the dump, so try the layout it
    * names first.  Note that this isn't always available (the a650-family
    * and a7xx kernel paths return early without stuffing it), so treat it as
    * a hint and validate it like any other candidate.
    */
   if (!is_bv && (0x2000 < sizedwords)) {
      for (unsigned i = 0; i < ARRAY_SIZE(mempool_layouts); i++) {
         if (mempool[0x2000] == mempool_layouts[i].num_blocks)
            order[n++] = i;
      }
   }

   order[n++] = dflt;

   for (unsigned i = 0; i < ARRAY_SIZE(mempool_layouts); i++)
      order[n++] = i;

   int best = -1;
   int best_matches = -1;

   for (unsigned i = 0; i < n; i++) {
      int matches = validate_layout(mempool, sizedwords,
                                    &mempool_layouts[order[i]], num_queues);

      if (matches == (int)num_queues)
         return order[i];

      if (matches > best_matches) {
         best_matches = matches;
         best = order[i];
      }
   }

   if (best < 0)
      best = dflt;

   mesa_logw_once("\tWARNING: could not determine CP mem pool layout, assuming '%s'\n",
                  mempool_layouts[best].name);

   return best;
}

void
dump_cp_mem_pool(uint32_t *mempool, uint32_t sizedwords, bool is_bv)
{
   /* The mem pool is a shared pool of memory used for storing in-flight
    * register writes. There are 6 different queues, one for each
    * cluster. Writing to $data (or for some special registers, $addr)
    * pushes data onto the appropriate queue, and each queue is pulled
    * from by the appropriate cluster. The queues are thus written to
    * in-order, but may be read out-of-order.
    *
    * The queues are conceptually divided into 128-bit "chunks", and the
    * read and write pointers are in units of chunks.  These chunks are
    * organized internally into 8-chunk (or on certain cut-down parts,
    * 4-chunk) "blocks", and memory is allocated dynamically in terms of
    * blocks. Each queue is represented as a singly-linked list of blocks, as
    * well as 3-bit start/end chunk pointers that point within the first/last
    * block.  The next pointers are located in a separate array, rather than
    * inline.
    */

   /* Number of queues or clusters. On a7xx the clusters are shuffled a bit to
    * increase the count by one, but BV seems to not use the last cluster.
    */
   const unsigned num_queues = (is_a6xx() || is_bv) ? 6 : 7;

   /* Some parts have a smaller mem pool, which shifts the offsets of the
    * next-pointers array and the per-queue state.  Detect which layout this
    * dump uses rather than assuming, since guessing wrong makes the queue
    * walk follow an unrelated (and possibly cyclic) linked list:
    */
   const enum mempool_size mempool_size =
      detect_mempool_size(mempool, sizedwords, is_bv, num_queues);
   const struct mempool_layout *layout = &mempool_layouts[mempool_size];

   if (verbose)
      printf("\tmem pool layout: %s\n", layout->name);

   struct mempool_data1 data1;
   memcpy(&data1, &mempool[layout->data1_offset], sizeof(data1));

   STATIC_ASSERT(sizeof(struct mempool_data2) == 4 * 0x60);
   STATIC_ASSERT(offsetof(struct mempool_data2, queue_sizes) == 4 * 0x18);
   struct mempool_data2 data2;
   memcpy(&data2, &mempool[layout->data2_offset], sizeof(data2));

   printf("\tdata2:\n");
   dump_hex_ascii(&mempool[layout->data2_offset], sizeof(data2), 1);

   /* These seem to be some kind of counter of allocated/deallocated blocks */
   if (verbose) {
      printf("\tunk0: %x\n", data1.unk0);
      printf("\tunk1: %x\n", data1.unk1);
      printf("\tunk3: %x\n", data1.unk3);
      printf("\tunk4: %x\n\n", data1.unk4);
   }

   for (unsigned queue = 0; queue < num_queues; queue++) {
      const char *cluster_names_a6xx[6] = {"FE",   "SP_VS", "PC_VS",
                                           "GRAS", "SP_PS", "PS"};
      const char *cluster_names_a7xx[7] = {"FE",   "SP_VS", "PC_VS",
                                           "GRAS", "SP_PS", "VPC_PS", "PS"};
      printf("\tCLUSTER_%s:\n\n",
             is_a6xx() ? cluster_names_a6xx[queue] : cluster_names_a7xx[queue]);

      int calculated_queue_size =
         walk_queue(mempool, layout, &data1, &data2, queue, true);

      if (calculated_queue_size < 0) {
         mesa_logw_once("\t\tINVALID QUEUE STATE, SKIPPING!\n");
      } else if ((uint32_t)calculated_queue_size != data2.queue_sizes[queue]) {
         printf("\t\tCALCULATED SIZE %d DOES NOT MATCH!\n",
                calculated_queue_size);
      }
      printf("\n");
   }
}

