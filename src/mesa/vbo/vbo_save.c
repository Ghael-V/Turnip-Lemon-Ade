/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Keith Whitwell <keithw@vmware.com>
 */


#include "main/arrayobj.h"
#include "main/bufferobj.h"

#include "util/u_memory.h"
#include "gallium/include/pipe/p_screen.h"

#include "vbo_private.h"


/**
 * Called at context creation time.
 */
void vbo_save_init( struct gl_context *ctx )
{
   struct vbo_context *vbo = vbo_context(ctx);
   struct vbo_save_context *save = &vbo->save;
   struct pipe_screen *screen = ctx->screen;

   vbo_save_api_init( save );

   save->restore_attrib_after_draw = true;

   ctx->Driver.CurrentSavePrimitive = PRIM_OUTSIDE_BEGIN_END;

   save->free_bo_pool_n_entries = CLAMP((screen->caps.video_memory * 1024 * 1024lu * 0.1) / VBO_SAVE_BUFFER_SIZE, 1, 128);
}


void vbo_save_destroy( struct gl_context *ctx )
{
   struct vbo_context *vbo = vbo_context(ctx);
   struct vbo_save_context *save = &vbo->save;

   if (save->prim_store) {
      free(save->prim_store->prims);
      FREE(save->prim_store);
      save->prim_store = NULL;
   }
   if (save->vertex_store) {
      free(save->vertex_store->buffer_in_ram);
      FREE(save->vertex_store);
      save->vertex_store = NULL;
   }

   free(save->copied.buffer);

   if (save->free_bo_pool) {
      for (int i = 0; i < save->free_bo_pool_n_entries; i++)
         _vbo_release_pool_entry(ctx, &save->free_bo_pool[i]);
      free(save->free_bo_pool);
   }
}
