/* Copyright © 2023 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef __GENX_HELPERS_H__
#define __GENX_HELPERS_H__

#include "util/u_math.h"

#ifndef __gen_validate_value
#define __gen_validate_value(x)
#endif

#ifndef __intel_field_functions
#define __intel_field_functions
#endif

#ifndef __gen_address_type
#error #define __gen_address_type before including this file
#endif

#ifndef __gen_user_data
#error #define __gen_combine_address before including this file
#endif

#ifndef CONST
#define CONST const
#endif

static inline __attribute__((always_inline)) uint64_t
__gen_offset(uint64_t v, ASSERTED uint32_t start, ASSERTED uint32_t end)
{
   __gen_validate_value(v);
#ifndef NDEBUG
   uint64_t mask = (UINT64_MAX >> (64 - (end - start + 1))) << start;

   assert((v & ~mask) == 0);
#endif

   return v;
}

static inline __attribute__((always_inline)) uint64_t
__gen_offset_nonzero(uint64_t v, uint32_t start, uint32_t end)
{
   assert(v != 0ull);
   return __gen_offset(v, start, end);
}

static inline __attribute__((always_inline)) uint64_t
__gen_address(__gen_user_data *data, void *location,
              __gen_address_type address, uint32_t delta,
              __attribute__((unused)) uint32_t start, uint32_t end)
{
   uint64_t addr_u64 = __gen_combine_address(data, location, address, delta);
   if (end == 31) {
      return addr_u64;
   } else if (end < 63) {
      const unsigned shift = 63 - end;
      return (addr_u64 << shift) >> shift;
   } else {
      return addr_u64;
   }
}

static inline uint64_t
__gen_unpack_uint(CONST uint32_t *restrict cl, uint32_t start, uint32_t end)
{
   uint64_t val = 0;
   const int width = end - start + 1;
   const uint64_t mask =
      (width == 64) ? ~((uint64_t)0) : ((uint64_t)1 << width) - 1;

   for (unsigned word = start / 32; word < (end / 32) + 1; word++) {
      val |= ((uint64_t)cl[word]) << ((word - start / 32) * 32);
   }

   return (val >> (start % 32)) & mask;
}

static inline uint64_t
__gen_unpack_sint(CONST uint32_t *restrict cl, uint32_t start, uint32_t end)
{
   int size = end - start + 1;
   int64_t val = __gen_unpack_uint(cl, start, end);

   return util_sign_extend(val, size);
}

#define __gen_unpack_float(x, y, z) uif(__gen_unpack_uint(x, y, z))

#endif /* __GENX_HELPERS_H__ */
