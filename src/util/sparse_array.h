/*
 * Copyright © 2019 Intel Corporation
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

#ifndef _UTIL_SPARSE_ARRAY_H
#define _UTIL_SPARSE_ARRAY_H

#include <stdint.h>

#include "c11/threads.h"
#include "macros.h"
#include "u_atomic.h"
#include "u_math.h"

#ifdef __cplusplus
extern "C" {
#endif

struct util_sparse_array_node;

/** A thread-safe automatically growing sparse array data structure
 *
 * This data structure has the following very nice properties:
 *
 *  1. Accessing an element is basically constant time.  Technically, it's
 *     O(log_b n) where the base b is the node size and n is the maximum
 *     index.  However, node sizes are expected to be fairly large and the
 *     index is a uint64_t so, if your node size is 256, it's O(8).
 *
 *  2. The data stored in the array is never moved in memory.  Instead, the
 *     data structure only ever grows and new nodes are added as-needed.  This
 *     means it's safe to store a pointer to something stored in the sparse
 *     array without worrying about a realloc invalidating it.
 *
 *  3. The data structure is thread-safe.  No guarantees are made about the
 *     data stored in the sparse array but it is safe to call
 *     util_sparse_array_get(arr, idx) from as many threads as you'd like and
 *     we guarantee that two calls to util_sparse_array_get(arr, idx) with the
 *     same array and index will always return the same pointer regardless
 *     contention between threads.
 *
 *  4. The data structure is lock-free.  All manipulations of the tree are
 *     done by a careful use of atomics to maintain thread safety and no locks
 *     are ever taken other than those taken implicitly by calloc().  If no
 *     allocation is required, util_sparse_array_get(arr, idx) does a simple
 *     walk over the tree should be efficient even in the case where many
 *     threads are accessing the sparse array at once.
 */
struct util_sparse_array {
   size_t elem_size;
   unsigned node_size_log2;

   uintptr_t root;
};

void util_sparse_array_init(struct util_sparse_array *arr,
                            size_t elem_size, size_t node_size);

void util_sparse_array_finish(struct util_sparse_array *arr);

/* Aligning our allocations to 64 has two advantages:
 *
 *  1. On x86 platforms, it means that they are cache-line aligned so we
 *     reduce the likelihood that one of our allocations shares a cache line
 *     with some other allocation.
 *
 *  2. It lets us use the bottom 6 bits of the pointer to store the tree level
 *     of the node so we can avoid some pointer indirections.
 */
#define UTIL_SPARSE_ARRAY_NODE_ALIGN 64
#define UTIL_SPARSE_ARRAY_NODE_LEVEL_MASK \
   ((uintptr_t)UTIL_SPARSE_ARRAY_NODE_ALIGN - 1)

static inline void *
_util_sparse_array_node_data(uintptr_t handle)
{
   return (void *)(handle & ~UTIL_SPARSE_ARRAY_NODE_LEVEL_MASK);
}

static inline unsigned
_util_sparse_array_node_level(uintptr_t handle)
{
   return handle & UTIL_SPARSE_ARRAY_NODE_LEVEL_MASK;
}

/* Out-of-line part of util_sparse_array_get() which allocates any missing
 * nodes on the way to the element.
 */
void *_util_sparse_array_get_slow(struct util_sparse_array *arr, uint64_t idx);

/* This only walks nodes which already exist, which is the common case, and
 * calls _util_sparse_array_get_slow() for everything else.
 */
static inline void *
util_sparse_array_get(struct util_sparse_array *arr, uint64_t idx)
{
   const unsigned node_size_log2 = arr->node_size_log2;
   const uint64_t node_mask = (1ull << node_size_log2) - 1;
   uintptr_t node = p_atomic_read(&arr->root);

   if (unlikely(!node))
      return _util_sparse_array_get_slow(arr, idx);

   unsigned level = _util_sparse_array_node_level(node);

   /* The root is not deep enough to contain this index yet. */
   if (unlikely((idx >> (level * node_size_log2)) > node_mask))
      return _util_sparse_array_get_slow(arr, idx);

   while (level > 0) {
      uintptr_t *children = (uintptr_t *)_util_sparse_array_node_data(node);
      const uint64_t child_idx = (idx >> (level * node_size_log2)) & node_mask;

      node = p_atomic_read(&children[child_idx]);
      if (unlikely(!node))
         return _util_sparse_array_get_slow(arr, idx);

      level = _util_sparse_array_node_level(node);
   }

   return (char *)_util_sparse_array_node_data(node) +
          (idx & node_mask) * arr->elem_size;
}

void util_sparse_array_validate(struct util_sparse_array *arr);

/** A thread-safe free list for use with struct util_sparse_array
 *
 * This data structure provides an easy way to manage a singly linked list of
 * "free" elements backed by a util_sparse_array.  The list supports only two
 * operations: push and pop both of which are thread-safe and lock-free.  T
 */
struct util_sparse_array_free_list
{
   /** Head of the list
    *
    * The bottom 64 bits of this value are the index to the next free element
    * or the sentinel value if the list is empty.
    *
    * We want this element to be 8-byte aligned.  Otherwise, the performance
    * of atomic operations on it will be aweful on 32-bit platforms.
    */
   alignas(8) uint64_t head;

   /** The array backing this free list */
   struct util_sparse_array *arr;

   /** Sentinel value to indicate the end of the list
    *
    * This value must never be passed into util_sparse_array_free_list_push.
    */
   uint32_t sentinel;

   /** Offset into the array element at which to find the "next" value
    *
    * The assumption is that there is some uint32_t "next" value embedded in
    * the array element for use in the free list.  This is its offset.
    */
   uint32_t next_offset;
};

void util_sparse_array_free_list_init(struct util_sparse_array_free_list *fl,
                                      struct util_sparse_array *arr,
                                      uint32_t sentinel,
                                      uint32_t next_offset);

void util_sparse_array_free_list_push(struct util_sparse_array_free_list *fl,
                                      uint32_t *items, unsigned num_items);

uint32_t util_sparse_array_free_list_pop_idx(struct util_sparse_array_free_list *fl);
void *util_sparse_array_free_list_pop_elem(struct util_sparse_array_free_list *fl);

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* _UTIL_SPARSE_ARRAY_H */
