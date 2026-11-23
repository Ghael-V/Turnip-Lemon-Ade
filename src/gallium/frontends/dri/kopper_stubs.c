/*  SPDX-License-Identifier: MIT */

#include "dri_drawable.h"
#include "dri_screen.h"
#include "dri_util.h"

int64_t
kopperSwapBuffers(struct dri_drawable *dPriv, uint32_t flush_flags, int nrects, const int *rects)
{
   return 0;
}

void
kopperSetSwapInterval(struct dri_drawable *dPriv, int interval)
{
}

int
kopperQueryBufferAge(struct dri_drawable *dPriv)
{
   return 0;
}

void
kopperQuerySurfaceSize(struct dri_drawable *drawable, int *width, int *height)
{
   *width = *height = 1;
}

int
kopperGetSyncValues(struct dri_drawable *drawable, int64_t target_msc, int64_t divisor,
                    int64_t remainder, int64_t *ust, int64_t *msc, int64_t *sbc)
{
   return 0;
}

struct pipe_screen *
kopper_init_screen(struct dri_screen *screen, bool driver_name_is_inferred)
{
   return NULL;
}

void
kopper_init_drawable(struct dri_drawable *drawable, bool isPixmap, int alphaBits)
{
}

void
kopper_destroy_drawable(struct dri_drawable *drawable)
{
}

void
kopper_allocate_textures(struct dri_context *ctx,
                         struct dri_drawable *drawable,
                         const enum st_attachment_type *statts,
                         unsigned statts_count)
{
}

void
kopper_update_drawable_info(struct dri_drawable *drawable)
{
}

bool
kopper_flush_frontbuffer(struct dri_context *ctx,
                         struct dri_drawable *drawable,
                         enum st_attachment_type statt)
{
   return false;
}

void
kopper_update_tex_buffer(struct dri_drawable *drawable,
                         struct dri_context *ctx,
                         struct pipe_resource *res)
{
}
