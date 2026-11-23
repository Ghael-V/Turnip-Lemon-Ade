/*
 * SPDX-License-Identifier: MIT
 */

#ifndef ZINK_LIST_RESTART_H
#define ZINK_LIST_RESTART_H

#include "pipe/p_state.h"

struct zink_context;

#ifdef __cplusplus
extern "C" {
#endif

bool
zink_compact_list_restart(struct zink_context *ctx,
                          const struct pipe_draw_info *dinfo,
                          const struct pipe_draw_start_count_bias *draw,
                          struct pipe_draw_info *out_info,
                          struct pipe_draw_indirect_info *out_indirect,
                          struct pipe_draw_start_count_bias *out_draw);

bool
zink_list_restart_init(struct zink_context *ctx);

void
zink_list_restart_deinit(struct zink_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
