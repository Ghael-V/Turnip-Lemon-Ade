/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef INTEL_HANG_REPLAY_RT_H
#define INTEL_HANG_REPLAY_RT_H

#include <stddef.h>

struct intel_device_info;
struct util_dynarray;
struct intel_hang_dump_block_exec;

void intel_hang_replay_validate_rt(int file_fd, size_t file_size,
                                   const struct intel_device_info *devinfo,
                                   struct util_dynarray *buffers,
                                   const struct intel_hang_dump_block_exec *exec);

#endif
