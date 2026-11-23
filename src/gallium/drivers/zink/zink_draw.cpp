#include "zink_batch.h"
#include "zink_compiler.h"
#include "zink_context.h"
#include "zink_descriptors.h"
#include "zink_program.h"
#include "zink_program_state.hpp"
#include "zink_query.h"
#include "zink_resource.h"
#include "zink_screen.h"
#include "zink_state.h"
#include "zink_inlines.h"
#include "zink_list_restart.h"

#include "util/hash_table.h"
#include "util/u_cpu_detect.h"
#include "util/u_debug.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_prim.h"
#include "util/u_prim_restart.h"

static void
zink_emit_xfb_counter_barrier(struct zink_context *ctx)
{
   for (unsigned i = 0; i < ctx->num_so_targets; i++) {
      struct zink_so_target *t = zink_so_target(ctx->so_targets[i]);
      if (!t)
         continue;
      struct zink_resource *res = zink_resource(t->counter_buffer);
      VkAccessFlags access = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
      if (t->counter_buffer_valid) {
         /* Between the pause and resume there needs to be a memory barrier for the counter buffers
          * with a source access of VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT
          * at pipeline stage VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT
          * to a destination access of VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT
          * at pipeline stage VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT.
          *
          * - from VK_EXT_transform_feedback spec
          */
         access |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;
         stage |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      }
      zink_screen(ctx->base.screen)->buffer_barrier(ctx, res, access, stage);
      if (!ctx->unordered_blitting)
         zink_resource_disable_unordered(res, false);
   }
}

static void
zink_emit_stream_output_targets(struct pipe_context *pctx)
{
   struct zink_context *ctx = zink_context(pctx);
   VkBuffer buffers[PIPE_MAX_SO_BUFFERS] = {0};
   VkDeviceSize buffer_offsets[PIPE_MAX_SO_BUFFERS] = {0};
   VkDeviceSize buffer_sizes[PIPE_MAX_SO_BUFFERS] = {0};

   for (unsigned i = 0; i < ctx->num_so_targets; i++) {
      struct zink_so_target *t = (struct zink_so_target *)ctx->so_targets[i];
      if (!t) {
         /* no need to reference this or anything */
         buffers[i] = zink_resource(ctx->dummy_xfb_buffer)->obj->buffer;
         buffer_offsets[i] = 0;
         buffer_sizes[i] = sizeof(uint8_t);
         continue;
      }
      struct zink_resource *res = zink_resource(t->base.buffer);
      if (!res->so_valid)
         /* resource has been rebound */
         t->counter_buffer_valid = false;
      buffers[i] = res->obj->buffer;
      zink_batch_reference_resource_rw(ctx, res, true);
      buffer_offsets[i] = t->base.buffer_offset;
      buffer_sizes[i] = t->base.buffer_size;
      res->so_valid = true;
      if (!ctx->unordered_blitting) {
         zink_resource_disable_unordered(res, true);
         res->obj->access = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
         res->obj->access_stage = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
      }
      util_range_add(t->base.buffer, &res->valid_buffer_range, t->base.buffer_offset,
                     t->base.buffer_offset + t->base.buffer_size);
   }

   VKCTX(CmdBindTransformFeedbackBuffersEXT)(ctx->bs->cmdbuf, 0, ctx->num_so_targets,
                                                 buffers, buffer_offsets,
                                                 buffer_sizes);
   ctx->dirty_so_targets = false;
}

ALWAYS_INLINE static void
check_buffer_barrier(struct zink_context *ctx, struct pipe_resource *pres, VkAccessFlags flags, VkPipelineStageFlags pipeline)
{
   struct zink_resource *res = zink_resource(pres);
   /* fastpath for viewperf */
   if (res->obj->access != flags || res->obj->access_stage != pipeline)
      zink_buffer_barrier(ctx, res, flags, pipeline);
   if (!ctx->unordered_blitting)
      zink_resource_disable_unordered(res, false);
}

/* with COPY, the buffers come with the draw and are stored in ctx->vertex_buffers in the same pass */
template <bool COPY>
ALWAYS_INLINE static void
barrier_unowned_vertex_buffers(struct zink_context *ctx, const struct pipe_vertex_buffer *buffers, unsigned count)
{
   const bool unordered_blitting = ctx->unordered_blitting;
   for (unsigned i = 0; i < count; i++) {
      if (COPY)
         ctx->vertex_buffers[i] = buffers[i];
      struct zink_resource *res = zink_resource(buffers[i].buffer.resource);
      if (!res)
         continue;
      if (res->obj->access != VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT ||
          res->obj->access_stage != VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
         zink_buffer_barrier(ctx, res, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
      if (!unordered_blitting)
         zink_resource_disable_unordered(res, false);
      zink_batch_resource_usage_set(ctx->bs, res, false, true);
   }
}

ALWAYS_INLINE static void
update_drawid(struct zink_context *ctx, unsigned draw_id)
{
   VKCTX(CmdPushConstants)(ctx->bs->cmdbuf, ctx->curr_program->base.layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                      offsetof(struct zink_gfx_push_constant, draw_id), sizeof(unsigned),
                      &draw_id);
}

ALWAYS_INLINE static void
draw_indexed_need_index_buffer_unref(struct zink_context *ctx,
             const struct pipe_draw_info *dinfo,
             const struct pipe_draw_start_count_bias *draws,
             unsigned num_draws,
             unsigned draw_id,
             bool needs_drawid)
{
   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;
   if (needs_drawid && dinfo->increment_draw_id) {
      for (unsigned i = 0; i < num_draws; i++) {
         update_drawid(ctx, draw_id);
         VKCTX(CmdDrawIndexed)(cmdbuf,
            draws[i].count, dinfo->instance_count,
            0, draws[i].index_bias, dinfo->start_instance);
         draw_id++;
      }
   } else {
      if (needs_drawid)
         update_drawid(ctx, draw_id);
      for (unsigned i = 0; i < num_draws; i++)
         VKCTX(CmdDrawIndexed)(cmdbuf,
            draws[i].count, dinfo->instance_count,
            0, draws[i].index_bias, dinfo->start_instance);

   }
}

template <zink_multidraw HAS_MULTIDRAW>
ALWAYS_INLINE static void
draw_indexed(struct zink_context *ctx,
             const struct pipe_draw_info *dinfo,
             const struct pipe_draw_start_count_bias *draws,
             unsigned num_draws,
             unsigned draw_id,
             bool needs_drawid)
{
   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;
   if (needs_drawid && dinfo->increment_draw_id) {
      for (unsigned i = 0; i < num_draws; i++) {
         update_drawid(ctx, draw_id);
         VKCTX(CmdDrawIndexed)(cmdbuf,
            draws[i].count, dinfo->instance_count,
            draws[i].start, draws[i].index_bias, dinfo->start_instance);
         draw_id++;
      }
   } else {
      if (needs_drawid)
         update_drawid(ctx, draw_id);
      if (HAS_MULTIDRAW && num_draws > 1) {
         VKCTX(CmdDrawMultiIndexedEXT)(cmdbuf, num_draws, (const VkMultiDrawIndexedInfoEXT*)draws,
                                       dinfo->instance_count,
                                       dinfo->start_instance, sizeof(struct pipe_draw_start_count_bias),
                                       dinfo->index_bias_varies ? NULL : &draws[0].index_bias);
      } else {
         for (unsigned i = 0; i < num_draws; i++)
            VKCTX(CmdDrawIndexed)(cmdbuf,
               draws[i].count, dinfo->instance_count,
               draws[i].start, draws[i].index_bias, dinfo->start_instance);
      }
   }
}

template <zink_multidraw HAS_MULTIDRAW>
ALWAYS_INLINE static void
draw(struct zink_context *ctx,
     const struct pipe_draw_info *dinfo,
     const struct pipe_draw_start_count_bias *draws,
     unsigned num_draws,
     unsigned draw_id,
     bool needs_drawid)
{
   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;
   if (needs_drawid && dinfo->increment_draw_id) {
      for (unsigned i = 0; i < num_draws; i++) {
         update_drawid(ctx, draw_id);
         VKCTX(CmdDraw)(cmdbuf, draws[i].count, dinfo->instance_count, draws[i].start, dinfo->start_instance);
         draw_id++;
      }
   } else {
      if (needs_drawid)
         update_drawid(ctx, draw_id);
      if (HAS_MULTIDRAW && num_draws > 1)
         VKCTX(CmdDrawMultiEXT)(cmdbuf, num_draws, (const VkMultiDrawInfoEXT*)draws,
                                dinfo->instance_count, dinfo->start_instance,
                                sizeof(struct pipe_draw_start_count_bias));
      else {
         for (unsigned i = 0; i < num_draws; i++)
            VKCTX(CmdDraw)(cmdbuf, draws[i].count, dinfo->instance_count, draws[i].start, dinfo->start_instance);

      }
   }
}

template <zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED>
static bool
update_gfx_pipeline(struct zink_context *ctx, struct zink_batch_state *bs, enum mesa_prim mode, bool *pipeline_changed_out)
{
   VkPipeline prev_pipeline = ctx->gfx_pipeline_state.pipeline;
   const struct zink_screen *screen = zink_screen(ctx->base.screen);
   if (BATCH_CHANGED && ctx->curr_program) {
      if (zink_gfx_program_needs_replacement(ctx->curr_program))
         zink_gfx_program_queue_replacement(ctx, ctx->curr_program);
      else if (zink_gfx_program_has_replacement(ctx->curr_program))
         ctx->gfx_dirty = true;
      /* a pipeline which is drawn with unchanged state is not looked up again, so count its use in this batch here */
      struct zink_gfx_pipeline_cache_entry *last_pipeline = ctx->curr_program->last_pipeline[ctx->gfx_pipeline_state.idx];
      if (last_pipeline && unlikely(last_pipeline->compile_pending))
         zink_gfx_program_compile_queue(ctx, last_pipeline);
      if (zink_gfx_program_adopt_draw_variant(ctx->curr_program))
         ctx->gfx_dirty = true;
      else
         zink_gfx_program_queue_draw_variant(ctx, ctx->curr_program);
   }
   bool shaders_changed = ctx->gfx_dirty || ctx->dirty_gfx_stages;
   if (screen->optimal_keys && !ctx->is_generated_gs_bound)
      zink_gfx_program_update_optimal(ctx);
   else
      zink_gfx_program_update(ctx);
   bool pipeline_changed = false;
   VkPipeline pipeline = VK_NULL_HANDLE;
   if (!ctx->curr_program->base.uses_shobj) {
      if (screen->info.have_EXT_graphics_pipeline_library)
         pipeline = zink_get_gfx_pipeline<DYNAMIC_STATE, true, false>(ctx, ctx->curr_program, &ctx->gfx_pipeline_state, mode);
      else
         pipeline = zink_get_gfx_pipeline<DYNAMIC_STATE, false, false>(ctx, ctx->curr_program, &ctx->gfx_pipeline_state, mode);
      /* creation failed and was logged there; the caller skips the draw */
      if (unlikely(pipeline == VK_NULL_HANDLE))
         return false;
      pipeline_changed = prev_pipeline != pipeline || ctx->shobj_draw;
      if (BATCH_CHANGED || pipeline_changed)
         VKCTX(CmdBindPipeline)(bs->cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      ctx->shobj_draw = false;
   } else {
      if (BATCH_CHANGED || shaders_changed || !ctx->shobj_draw) {
         VkShaderStageFlagBits stages[] = {
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
         };
         /* always rebind all stages */
         VKCTX(CmdBindShadersEXT)(bs->cmdbuf, ZINK_GFX_SHADER_COUNT, stages, ctx->curr_program->objects);
         if (screen->info.have_EXT_mesh_shader) {
            /* must always unbind mesh stages */
            VkShaderStageFlagBits mesh_stages[] = {
               VK_SHADER_STAGE_TASK_BIT_EXT,
               VK_SHADER_STAGE_MESH_BIT_EXT,
            };
            VkShaderEXT mesh[2] = {VK_NULL_HANDLE};
            VKCTX(CmdBindShadersEXT)(bs->cmdbuf, ARRAY_SIZE(mesh_stages), mesh_stages, mesh);
         }
         VKCTX(CmdSetDepthBiasEnable)(bs->cmdbuf, VK_TRUE);
         VKCTX(CmdSetTessellationDomainOriginEXT)(bs->cmdbuf, VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT);
         VKCTX(CmdSetRasterizationStreamEXT)(bs->cmdbuf, 0);
         pipeline_changed = true;
      }
      ctx->shobj_draw = true;
   }
   *pipeline_changed_out = pipeline_changed;
   return true;
}

static enum mesa_prim
zink_prim_type(const struct zink_context *ctx,
               const struct pipe_draw_info *dinfo)
{
   if (ctx->gfx_pipeline_state.shader_rast_prim != MESA_PRIM_COUNT)
      return ctx->gfx_pipeline_state.shader_rast_prim;

   return u_reduced_prim((enum mesa_prim)dinfo->mode);
}

static enum mesa_prim
zink_rast_prim(const struct zink_context *ctx,
               const struct pipe_draw_info *dinfo)
{
   enum mesa_prim prim_type = zink_prim_type(ctx, dinfo);
   assert(prim_type != MESA_PRIM_COUNT);

   if (prim_type == MESA_PRIM_TRIANGLES &&
       ctx->rast_state->base.fill_front != PIPE_POLYGON_MODE_FILL) {
      switch(ctx->rast_state->base.fill_front) {
      case PIPE_POLYGON_MODE_POINT:
         return MESA_PRIM_POINTS;
      case PIPE_POLYGON_MODE_LINE:
         return MESA_PRIM_LINES;
      default:
         UNREACHABLE("unexpected polygon mode");
      }
   }

   return prim_type;
}

/* set the dynamic state of the given atoms on the cmdbuf */
template <zink_dynamic_state DYNAMIC_STATE>
ALWAYS_INLINE static void
emit_dynamic_atoms(struct zink_context *ctx, uint32_t atoms, unsigned num_viewports)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   struct zink_batch_state *bs = ctx->bs;
   struct zink_rasterizer_state *rast_state = ctx->rast_state;
   struct zink_depth_stencil_alpha_state *dsa_state = ctx->dsa_state;

   u_foreach_bit(atom, atoms) {
      switch ((enum zink_atom)atom) {
      case ZINK_ATOM_VIEWPORT: {
         VkViewport viewports[PIPE_MAX_VIEWPORTS];
         for (unsigned i = 0; i < num_viewports; i++) {
            VkViewport viewport = {
               ctx->vp_state.viewport_states[i].translate[0] - ctx->vp_state.viewport_states[i].scale[0],
               ctx->vp_state.viewport_states[i].translate[1] - ctx->vp_state.viewport_states[i].scale[1],
               MAX2(ctx->vp_state.viewport_states[i].scale[0] * 2, 1),
               ctx->vp_state.viewport_states[i].scale[1] * 2,
               CLAMP(ctx->rast_state->base.clip_halfz ?
                     ctx->vp_state.viewport_states[i].translate[2] :
                     ctx->vp_state.viewport_states[i].translate[2] - ctx->vp_state.viewport_states[i].scale[2],
                     0, 1),
               CLAMP(ctx->vp_state.viewport_states[i].translate[2] + ctx->vp_state.viewport_states[i].scale[2],
                     0, 1)
            };
            if (!ctx->rast_state->base.half_pixel_center) {
                /* magic constant value from dxvk */
                float cf = 0.5f - (1.0f / 128.0f);
                viewport.x += cf;
                if (viewport.height < 0)
                   viewport.y += cf;
                else
                   viewport.y -= cf;
            }
            viewports[i] = viewport;
         }
         if (DYNAMIC_STATE != ZINK_NO_DYNAMIC_STATE)
            VKCTX(CmdSetViewportWithCount)(bs->cmdbuf, num_viewports, viewports);
         else
            VKCTX(CmdSetViewport)(bs->cmdbuf, 0, num_viewports, viewports);
         break;
      }
      case ZINK_ATOM_SCISSOR: {
         VkRect2D scissors[PIPE_MAX_VIEWPORTS];
         if (ctx->rast_state->base.scissor) {
            for (unsigned i = 0; i < num_viewports; i++) {
               scissors[i].offset.x = ctx->vp_state.scissor_states[i].minx;
               scissors[i].offset.y = ctx->vp_state.scissor_states[i].miny;
               scissors[i].extent.width = ctx->vp_state.scissor_states[i].maxx - ctx->vp_state.scissor_states[i].minx;
               scissors[i].extent.height = ctx->vp_state.scissor_states[i].maxy - ctx->vp_state.scissor_states[i].miny;
            }
         } else {
            for (unsigned i = 0; i < num_viewports; i++) {
               scissors[i].offset.x = 0;
               scissors[i].offset.y = 0;
               scissors[i].extent.width = ctx->fb_state.width;
               scissors[i].extent.height = ctx->fb_state.height;
            }
         }
         if (DYNAMIC_STATE != ZINK_NO_DYNAMIC_STATE)
            VKCTX(CmdSetScissorWithCount)(bs->cmdbuf, num_viewports, scissors);
         else
            VKCTX(CmdSetScissor)(bs->cmdbuf, 0, num_viewports, scissors);
         break;
      }
      case ZINK_ATOM_STENCIL_REF: {
         VKCTX(CmdSetStencilReference)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_BIT,
                                  ctx->stencil_ref.ref_value[0]);
         VKCTX(CmdSetStencilReference)(bs->cmdbuf, VK_STENCIL_FACE_BACK_BIT,
                                  ctx->stencil_ref.ref_value[1]);
         break;
      }
      case ZINK_ATOM_DSA: {
         if (DYNAMIC_STATE != ZINK_NO_DYNAMIC_STATE) {
            VKCTX(CmdSetDepthBoundsTestEnable)(bs->cmdbuf, dsa_state->hw_state.depth_bounds_test);
            if (dsa_state->hw_state.depth_bounds_test)
               VKCTX(CmdSetDepthBounds)(bs->cmdbuf,
                                   dsa_state->hw_state.min_depth_bounds,
                                   dsa_state->hw_state.max_depth_bounds);
            VKCTX(CmdSetDepthTestEnable)(bs->cmdbuf, dsa_state->hw_state.depth_test);
            VKCTX(CmdSetDepthCompareOp)(bs->cmdbuf, dsa_state->hw_state.depth_compare_op);
            VKCTX(CmdSetDepthWriteEnable)(bs->cmdbuf, dsa_state->hw_state.depth_write);
            VKCTX(CmdSetStencilTestEnable)(bs->cmdbuf, dsa_state->hw_state.stencil_test);
            if (dsa_state->hw_state.stencil_test) {
               VKCTX(CmdSetStencilOp)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_BIT,
                                             dsa_state->hw_state.stencil_front.failOp,
                                             dsa_state->hw_state.stencil_front.passOp,
                                             dsa_state->hw_state.stencil_front.depthFailOp,
                                             dsa_state->hw_state.stencil_front.compareOp);
               VKCTX(CmdSetStencilOp)(bs->cmdbuf, VK_STENCIL_FACE_BACK_BIT,
                                             dsa_state->hw_state.stencil_back.failOp,
                                             dsa_state->hw_state.stencil_back.passOp,
                                             dsa_state->hw_state.stencil_back.depthFailOp,
                                             dsa_state->hw_state.stencil_back.compareOp);
               if (dsa_state->base.stencil[1].enabled) {
                  VKCTX(CmdSetStencilWriteMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_BIT, dsa_state->hw_state.stencil_front.writeMask);
                  VKCTX(CmdSetStencilWriteMask)(bs->cmdbuf, VK_STENCIL_FACE_BACK_BIT, dsa_state->hw_state.stencil_back.writeMask);
                  VKCTX(CmdSetStencilCompareMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_BIT, dsa_state->hw_state.stencil_front.compareMask);
                  VKCTX(CmdSetStencilCompareMask)(bs->cmdbuf, VK_STENCIL_FACE_BACK_BIT, dsa_state->hw_state.stencil_back.compareMask);
               } else {
                  VKCTX(CmdSetStencilWriteMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_AND_BACK, dsa_state->hw_state.stencil_front.writeMask);
                  VKCTX(CmdSetStencilCompareMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_AND_BACK, dsa_state->hw_state.stencil_front.compareMask);
               }
            } else {
               VKCTX(CmdSetStencilWriteMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_AND_BACK, dsa_state->hw_state.stencil_front.writeMask);
               VKCTX(CmdSetStencilCompareMask)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_AND_BACK, dsa_state->hw_state.stencil_front.compareMask);
               VKCTX(CmdSetStencilOp)(bs->cmdbuf, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
            }
         }
         break;
      }
      case ZINK_ATOM_SAMPLE_LOCATIONS: {
         bool enabled = ctx->sample_locations_enabled || (!rast_state->base.multisample && ctx->gfx_pipeline_state.rast_samples);
         VKCTX(CmdSetSampleLocationsEnableEXT)(bs->cmdbuf, enabled);
         if (enabled) {
            VkSampleLocationsInfoEXT loc;
            zink_init_vk_sample_locations(ctx, &loc);
            VKCTX(CmdSetSampleLocationsEXT)(bs->cmdbuf, &loc);
         }
         break;
      }
      case ZINK_ATOM_BLEND_COLOR: {
         VKCTX(CmdSetBlendConstants)(bs->cmdbuf, ctx->blend_constants);
         break;
      }
      case ZINK_ATOM_RASTERIZER_DISCARD: {
         if (DYNAMIC_STATE >= ZINK_DYNAMIC_STATE2) {
            VKCTX(CmdSetRasterizerDiscardEnable)(bs->cmdbuf, ctx->gfx_pipeline_state.dyn_state2.rasterizer_discard);
         }
         break;
      }
      case ZINK_ATOM_RASTERIZER: {
         if (DYNAMIC_STATE != ZINK_NO_DYNAMIC_STATE) {
            VKCTX(CmdSetFrontFace)(bs->cmdbuf, (VkFrontFace)ctx->gfx_pipeline_state.dyn_state1.front_face);
            VKCTX(CmdSetCullMode)(bs->cmdbuf, ctx->gfx_pipeline_state.dyn_state1.cull_mode);
         }

         if (DYNAMIC_STATE >= ZINK_DYNAMIC_STATE3) {
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_STIPPLE))
               VKCTX(CmdSetLineStippleEXT)(bs->cmdbuf, rast_state->base.line_stipple_factor, rast_state->base.line_stipple_pattern);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_CLIP))
               VKCTX(CmdSetDepthClipEnableEXT)(bs->cmdbuf, rast_state->hw_state.depth_clip);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_CLAMP))
               VKCTX(CmdSetDepthClampEnableEXT)(bs->cmdbuf, rast_state->hw_state.depth_clamp);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_POLYGON))
               VKCTX(CmdSetPolygonModeEXT)(bs->cmdbuf, (VkPolygonMode)rast_state->hw_state.polygon_mode);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_HALFZ))
               VKCTX(CmdSetDepthClipNegativeOneToOneEXT)(bs->cmdbuf, !rast_state->hw_state.clip_halfz);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_PV))
               VKCTX(CmdSetProvokingVertexModeEXT)(bs->cmdbuf,
                                                   rast_state->hw_state.pv_last ?
                                                   VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT :
                                                   VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_LINE))
               VKCTX(CmdSetLineRasterizationModeEXT)(bs->cmdbuf, ctx->fb_float_cbuf ?
                                                     rast_state->dynamic_line_mode_float :
                                                     rast_state->dynamic_line_mode);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_RAST_STIPPLE_ON))
               VKCTX(CmdSetLineStippleEnableEXT)(bs->cmdbuf, rast_state->hw_state.line_stipple_enable);
            ctx->ds3_states &= ~BITFIELD_RANGE(ZINK_DS3_RAST_STIPPLE, ZINK_DS3_RAST_STIPPLE_ON - ZINK_DS3_RAST_STIPPLE + 1);
         }
         break;
      }
      case ZINK_ATOM_SAMPLE_MASK: {
         if (screen->have_full_ds3) {
            VKCTX(CmdSetRasterizationSamplesEXT)(bs->cmdbuf, (VkSampleCountFlagBits)(ctx->gfx_pipeline_state.rast_samples + 1));
            VKCTX(CmdSetSampleMaskEXT)(bs->cmdbuf, (VkSampleCountFlagBits)(ctx->gfx_pipeline_state.rast_samples + 1), &ctx->gfx_pipeline_state.sample_mask);
         }
         break;
      }
      case ZINK_ATOM_BLEND: {
         if (ctx->gfx_pipeline_state.blend_state) {
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_A2C))
               VKCTX(CmdSetAlphaToCoverageEnableEXT)(bs->cmdbuf, ctx->gfx_pipeline_state.blend_state->alpha_to_coverage &&
                                                                           ctx->gfx_stages[MESA_SHADER_FRAGMENT]->info.outputs_written & BITFIELD_BIT(FRAG_RESULT_DATA0));
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_A21))
               VKCTX(CmdSetAlphaToOneEnableEXT)(bs->cmdbuf, ctx->gfx_pipeline_state.blend_state->alpha_to_one);
            if (ctx->fb_state.nr_cbufs) {
               if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_ON))
                  VKCTX(CmdSetColorBlendEnableEXT)(bs->cmdbuf, 0, ctx->fb_state.nr_cbufs, ctx->gfx_pipeline_state.blend_state->ds3.enables);
               if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_WRITE))
                  VKCTX(CmdSetColorWriteMaskEXT)(bs->cmdbuf, 0, ctx->fb_state.nr_cbufs, ctx->gfx_pipeline_state.blend_state->ds3.wrmask);
               if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_EQ))
                  VKCTX(CmdSetColorBlendEquationEXT)(bs->cmdbuf, 0, ctx->fb_state.nr_cbufs, ctx->gfx_pipeline_state.blend_state->ds3.eq);
            }
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_LOGIC_ON))
               VKCTX(CmdSetLogicOpEnableEXT)(bs->cmdbuf, ctx->gfx_pipeline_state.blend_state->logicop_enable);
            if (ctx->ds3_states & BITFIELD_BIT(ZINK_DS3_BLEND_LOGIC))
               VKCTX(CmdSetLogicOpEXT)(bs->cmdbuf, ctx->gfx_pipeline_state.blend_state->logicop_func);
            ctx->ds3_states &= ~BITFIELD_RANGE(ZINK_DS3_BLEND_A2C, ZINK_DS3_BLEND_LOGIC - ZINK_DS3_BLEND_A2C + 1);
         }
         break;
      }
      case ZINK_ATOM_DEPTH_BIAS:
         if (ctx->was_using_depth_bias)
            VKCTX(CmdSetDepthBias)(bs->cmdbuf, rast_state->offset_units, rast_state->offset_clamp, rast_state->offset_scale);
         else
            VKCTX(CmdSetDepthBias)(bs->cmdbuf, 0.0f, 0.0f, 0.0f);
         break;
      case ZINK_ATOM_LINE_WIDTH:
         VKCTX(CmdSetLineWidth)(bs->cmdbuf, rast_state->line_width);
         break;
      default:
         UNREACHABLE("not a dynamic state atom");
      }
   }
   ctx->dirty_atoms &= ~atoms;
}

/* emit the dynamic state atoms which are dirty, or all of them for a new cmdbuf, except for skip_atoms */
template <zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED>
ALWAYS_INLINE static void
emit_dynamic_state(struct zink_context *ctx, bool pipeline_changed, unsigned num_viewports, bool uses_shobj, uint32_t skip_atoms)
{
   if (ctx->depth_op_promoted && !ctx->can_promote_depth_op)
      zink_update_depth_state(ctx);

   uint32_t atoms = ctx->dirty_atoms & ZINK_DYNAMIC_STATE_ATOMS;
   if (BATCH_CHANGED) {
      atoms |= ZINK_DYNAMIC_STATE_ATOMS & ~BITFIELD_BIT(ZINK_ATOM_SAMPLE_LOCATIONS);
      if (zink_screen(ctx->base.screen)->base.caps.programmable_sample_locations || uses_shobj)
         atoms |= BITFIELD_BIT(ZINK_ATOM_SAMPLE_LOCATIONS);
   } else if (DYNAMIC_STATE == ZINK_NO_DYNAMIC_STATE && pipeline_changed) {
      atoms |= BITFIELD_BIT(ZINK_ATOM_VIEWPORT) | BITFIELD_BIT(ZINK_ATOM_SCISSOR);
   }
   /* the scissors are always set together with the viewports */
   if (atoms & BITFIELD_BIT(ZINK_ATOM_VIEWPORT))
      atoms |= BITFIELD_BIT(ZINK_ATOM_SCISSOR);
   atoms &= ~skip_atoms;
   if (atoms)
      emit_dynamic_atoms<DYNAMIC_STATE>(ctx, atoms, num_viewports);
}

template <zink_dynamic_state DYNAMIC_STATE>
static ALWAYS_INLINE void
bind_vertex_buffers(struct zink_context *ctx, const struct pipe_vertex_buffer *buffers)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);

   if (screen->info.have_KHR_device_address_commands)
      zink_bind_vertex_addresses(ctx, buffers);
   else if (DYNAMIC_STATE == ZINK_DYNAMIC_VERTEX_INPUT || DYNAMIC_STATE == ZINK_DYNAMIC_VERTEX_INPUT2 ||
         DYNAMIC_STATE == ZINK_NO_DYNAMIC_STATE || !ctx->gfx_pipeline_state.uses_dynamic_stride)
      zink_bind_vertex_buffers(ctx, buffers);
   else
      zink_bind_vertex_buffers_dynamic(ctx, buffers);
}

ALWAYS_INLINE static void
bind_index_buffer(struct zink_context *ctx, struct zink_resource *res, unsigned index_offset, unsigned index_size)
{
   const VkIndexType index_type[] = {
      VK_INDEX_TYPE_UINT32,
      VK_INDEX_TYPE_UINT8_EXT,
      VK_INDEX_TYPE_UINT16,
      VK_INDEX_TYPE_UINT32,
      VK_INDEX_TYPE_UINT32,
   };

   if (zink_screen(ctx->base.screen)->info.have_KHR_device_address_commands) {
      VkBindIndexBuffer3InfoKHR info;
      info.sType = VK_STRUCTURE_TYPE_BIND_INDEX_BUFFER_3_INFO_KHR;
      info.pNext = NULL;
      info.addressRange.address = res->obj->bda + index_offset;
      info.addressRange.size = res->size - index_offset;
      info.addressFlags = VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR |
                          VK_ADDRESS_COMMAND_TRANSFORM_FEEDBACK_BUFFER_USAGE_BIT_KHR;
      if (!res->is_sparse)
         info.addressFlags |= VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR;
      info.indexType = index_type[index_size];
      VKCTX(CmdBindIndexBuffer3KHR)(ctx->bs->cmdbuf, &info);
   } else {
      VKCTX(CmdBindIndexBuffer)(ctx->bs->cmdbuf, res->obj->buffer, index_offset, index_type[index_size]);
   }
}

/* state which is dirty without its atom being dirty is never set by a draw */
ALWAYS_INLINE static void
validate_atoms(const struct zink_context *ctx)
{
   assert(ctx->in_rp || zink_is_atom_dirty(ctx, ZINK_ATOM_RENDERPASS));
   assert(!ctx->rp_layout_changed || zink_is_atom_dirty(ctx, ZINK_ATOM_RENDERPASS));
   assert(!ctx->memory_barrier || zink_is_atom_dirty(ctx, ZINK_ATOM_MEMORY_BARRIER));
   assert(!ctx->num_so_targets || zink_is_atom_dirty(ctx, ZINK_ATOM_STREAMOUT));
   assert(!ctx->need_barriers[0]->entries || zink_is_atom_dirty(ctx, ZINK_ATOM_BARRIERS));
   assert((!ctx->vertices_query && list_is_empty(&ctx->primitives_generated_queries)) ||
          zink_is_atom_dirty(ctx, ZINK_ATOM_GS_QUERIES));
   assert(!(zink_debug & ZINK_DEBUG_SYNC) || zink_is_atom_dirty(ctx, ZINK_ATOM_DEBUG_SYNC));
   assert(!ctx->shader_reads_drawid || zink_is_atom_dirty(ctx, ZINK_ATOM_DRAW_ID));
   assert((!ctx->gfx_dirty && !ctx->dirty_gfx_stages) || zink_is_atom_dirty(ctx, ZINK_ATOM_PIPELINE));
   /* shader object programs never hash the pipeline state */
   assert(!ctx->gfx_pipeline_state.dirty || zink_is_atom_dirty(ctx, ZINK_ATOM_PIPELINE) ||
          (ctx->curr_program && ctx->curr_program->base.uses_shobj));
   /* a program without descriptors leaves these for the next program which has them */
   assert((!ctx->dd.push_state_changed[ZINK_PIPELINE_GFX] && !ctx->dd.state_changed[ZINK_PIPELINE_GFX]) ||
          zink_is_atom_dirty(ctx, ZINK_ATOM_DESCRIPTORS) || zink_is_atom_dirty(ctx, ZINK_ATOM_PIPELINE) ||
          !zink_program_has_descriptors(&ctx->curr_program->base));
}

/* offset_all_prims keeps the offset for lines and points, which the driver ignores in fill polygon mode,
 * but prim_is_rasterized must be false when prim is not what the rasterizer draws
 */
ALWAYS_INLINE static bool
zink_uses_depth_bias(const struct zink_rasterizer_state *rast_state, enum mesa_prim prim, bool prim_is_rasterized)
{
   return rast_state->offset_fill &&
          (prim == MESA_PRIM_TRIANGLES || (rast_state->offset_all_prims && prim_is_rasterized));
}

/* dirty the atoms for the inputs of this draw which differ from the last one */
template <zink_dynamic_state DYNAMIC_STATE>
ALWAYS_INLINE static void
mark_draw_atoms(struct zink_context *ctx, const struct pipe_draw_info *dinfo)
{
   const bool primitive_restart = DYNAMIC_STATE >= ZINK_DYNAMIC_STATE2 ?
                                  ctx->primitive_restart :
                                  ctx->gfx_pipeline_state.dyn_state2.primitive_restart;
   if (ctx->gfx_pipeline_state.gfx_prim_mode != dinfo->mode || primitive_restart != dinfo->primitive_restart)
      zink_mark_atom_dirty(ctx, ZINK_ATOM_PRIMITIVE);
   /* without dynamic vertex input, the vertex input state is part of the pipeline */
   if (DYNAMIC_STATE != ZINK_DYNAMIC_VERTEX_INPUT && DYNAMIC_STATE != ZINK_DYNAMIC_VERTEX_INPUT2 &&
       zink_is_atom_dirty(ctx, ZINK_ATOM_VERTEX_INPUT))
      zink_mark_atom_dirty(ctx, ZINK_ATOM_PIPELINE);
}

/* where the vertex buffers of a draw come from */
enum zink_draw_vbs {
   ZINK_VBS_BOUND, //set_vertex_buffers
   ZINK_VBS_UNOWNED, //passed to draw_vbo_buffers
   ZINK_VBS_VERTEX_STATE, //bound by draw_vertex_state
};

/* everything after the atoms: bind the buffers which come with the draw, and draw */
template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, zink_draw_vbs VBS, bool CLEAN>
ALWAYS_INLINE static void
zink_draw_finish(struct zink_context *ctx,
                 struct zink_batch_state *bs,
                 const struct pipe_draw_info *dinfo,
                 unsigned drawid_offset,
                 const struct pipe_draw_indirect_info *dindirect,
                 const struct pipe_draw_start_count_bias *draws,
                 unsigned num_draws,
                 struct zink_so_target *so_target,
                 struct pipe_resource *index_buffer,
                 unsigned index_offset,
                 unsigned index_size,
                 bool need_index_buffer_unref,
                 unsigned work_count,
                 bool have_streamout,
                 bool needs_drawid)
{
   struct pipe_context *pctx = &ctx->base;
   struct zink_screen *screen = zink_screen(pctx->screen);
   VkBuffer counter_buffers[PIPE_MAX_SO_BUFFERS];
   VkDeviceSize counter_buffer_offsets[PIPE_MAX_SO_BUFFERS];

   /* the buffers which come with the draw are not atoms */
   if (VBS == ZINK_VBS_UNOWNED) {
      bind_vertex_buffers<DYNAMIC_STATE>(ctx, ctx->vertex_buffers);
      zink_set_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS, false);
   }
   if (index_buffer)
      bind_index_buffer(ctx, zink_resource(index_buffer), index_offset, index_size);

   if (dinfo->primitive_restart && screen->info.have_EXT_primitive_restart_index)
      VKCTX(CmdSetPrimitiveRestartIndexEXT)(bs->cmdbuf, dinfo->restart_index);

   if (unlikely(have_streamout)) {
      if (ctx->dirty_so_targets)
         zink_emit_stream_output_targets(pctx);
      for (unsigned i = 0; i < ctx->num_so_targets; i++) {
         struct zink_so_target *t = zink_so_target(ctx->so_targets[i]);
         counter_buffers[i] = VK_NULL_HANDLE;
         if (t) {
            struct zink_resource *res = zink_resource(t->counter_buffer);
            t->stride = ctx->last_vertex_stage->xfb_stride[i];
            zink_batch_reference_resource_rw(ctx, res, true);
            if (!ctx->unordered_blitting)
               zink_resource_disable_unordered(res, true);
            if (t->counter_buffer_valid) {
               counter_buffers[i] = res->obj->buffer;
               counter_buffer_offsets[i] = t->counter_buffer_offset;
            }
         }
      }
      VKCTX(CmdBeginTransformFeedbackEXT)(bs->cmdbuf, 0, ctx->num_so_targets, counter_buffers, counter_buffer_offsets);
   }

   bool marker = false;
   if (!CLEAN && unlikely(zink_tracing)) {
      VkViewport viewport = {
         ctx->vp_state.viewport_states[0].translate[0] - ctx->vp_state.viewport_states[0].scale[0],
         ctx->vp_state.viewport_states[0].translate[1] - ctx->vp_state.viewport_states[0].scale[1],
         MAX2(ctx->vp_state.viewport_states[0].scale[0] * 2, 1),
         ctx->vp_state.viewport_states[0].scale[1] * 2,
         CLAMP(ctx->rast_state->base.clip_halfz ?
               ctx->vp_state.viewport_states[0].translate[2] :
               ctx->vp_state.viewport_states[0].translate[2] - ctx->vp_state.viewport_states[0].scale[2],
               0, 1),
         CLAMP(ctx->vp_state.viewport_states[0].translate[2] + ctx->vp_state.viewport_states[0].scale[2],
               0, 1)
      };
      if (ctx->blitting) {
         bool is_zs = util_format_is_depth_or_stencil(ctx->sampler_views[MESA_SHADER_FRAGMENT][0]->format);
         marker = zink_cmd_debug_marker_begin(ctx, VK_NULL_HANDLE, "u_blitter(%s->%s, %dx%d)",
                                              util_format_short_name(ctx->sampler_views[MESA_SHADER_FRAGMENT][0]->format),
                                              util_format_short_name((is_zs ? ctx->fb_state.zsbuf : ctx->fb_state.cbufs[0]).format),
                                              lround(viewport.width), lround(viewport.height));
      } else {
         marker = zink_cmd_debug_marker_begin(ctx, VK_NULL_HANDLE, "draw(%u cbufs|%s, %dx%d)",
                                              ctx->fb_state.nr_cbufs,
                                              ctx->fb_state.zsbuf.texture ? "zsbuf" : "",
                                              lround(viewport.width), lround(viewport.height));
      }
   }

   work_count += num_draws;
   if (index_size > 0) {
      if (dindirect && dindirect->buffer) {
         assert(num_draws == 1);
         if (needs_drawid)
            update_drawid(ctx, drawid_offset);
         struct zink_resource *indirect = zink_resource(dindirect->buffer);
         zink_batch_reference_resource_rw(ctx, indirect, false);
         if (dindirect->indirect_draw_count) {
             struct zink_resource *indirect_draw_count = zink_resource(dindirect->indirect_draw_count);
             zink_batch_reference_resource_rw(ctx, indirect_draw_count, false);
             VKCTX(CmdDrawIndexedIndirectCount)(bs->cmdbuf, indirect->obj->buffer, dindirect->offset,
                                                indirect_draw_count->obj->buffer, dindirect->indirect_draw_count_offset,
                                                dindirect->draw_count, dindirect->stride);
         } else
            VKCTX(CmdDrawIndexedIndirect)(bs->cmdbuf, indirect->obj->buffer, dindirect->offset, dindirect->draw_count, dindirect->stride);
      } else {
         if (need_index_buffer_unref)
            draw_indexed_need_index_buffer_unref(ctx, dinfo, draws, num_draws, drawid_offset, needs_drawid);
         else
            draw_indexed<HAS_MULTIDRAW>(ctx, dinfo, draws, num_draws, drawid_offset, needs_drawid);
      }
   } else {
      if (so_target && screen->info.tf_props.transformFeedbackDraw) {
         /* GTF-GL46.gtf40.GL3Tests.transform_feedback2.transform_feedback2_api attempts a bogus xfb
          * draw using a streamout target that has no data
          * to avoid hanging the gpu, reject any such draws
          */
         if (so_target->counter_buffer_valid) {
            if (needs_drawid)
               update_drawid(ctx, drawid_offset);
            zink_batch_reference_resource_rw(ctx, zink_resource(so_target->base.buffer), false);
            zink_batch_reference_resource_rw(ctx, zink_resource(so_target->counter_buffer), true);
            VKCTX(CmdDrawIndirectByteCountEXT)(bs->cmdbuf, dinfo->instance_count, dinfo->start_instance,
                                          zink_resource(so_target->counter_buffer)->obj->buffer, so_target->counter_buffer_offset, 0,
                                          MIN2(so_target->stride, screen->info.tf_props.maxTransformFeedbackBufferDataStride));
         }
      } else if (dindirect && dindirect->buffer) {
         assert(num_draws == 1);
         if (needs_drawid)
            update_drawid(ctx, drawid_offset);
         struct zink_resource *indirect = zink_resource(dindirect->buffer);
         zink_batch_reference_resource_rw(ctx, indirect, false);
         if (dindirect->indirect_draw_count) {
             struct zink_resource *indirect_draw_count = zink_resource(dindirect->indirect_draw_count);
             zink_batch_reference_resource_rw(ctx, indirect_draw_count, false);
             VKCTX(CmdDrawIndirectCount)(bs->cmdbuf, indirect->obj->buffer, dindirect->offset,
                                           indirect_draw_count->obj->buffer, dindirect->indirect_draw_count_offset,
                                           dindirect->draw_count, dindirect->stride);
         } else
            VKCTX(CmdDrawIndirect)(bs->cmdbuf, indirect->obj->buffer, dindirect->offset, dindirect->draw_count, dindirect->stride);
      } else {
         draw<HAS_MULTIDRAW>(ctx, dinfo, draws, num_draws, drawid_offset, needs_drawid);
      }
   }

   if (!CLEAN && unlikely(zink_tracing))
      zink_cmd_debug_marker_end(ctx, bs->cmdbuf, marker);

   if (unlikely(have_streamout)) {
      for (unsigned i = 0; i < ctx->num_so_targets; i++) {
         struct zink_so_target *t = zink_so_target(ctx->so_targets[i]);
         if (t) {
            counter_buffers[i] = zink_resource(t->counter_buffer)->obj->buffer;
            counter_buffer_offsets[i] = t->counter_buffer_offset;
            t->counter_buffer_valid = true;
         }
      }
      VKCTX(CmdEndTransformFeedbackEXT)(bs->cmdbuf, 0, ctx->num_so_targets, counter_buffers, counter_buffer_offsets);
   }

   ctx->last_work_was_compute = false;
   ctx->work_count = work_count;
   if (!ctx->pipeline_changed[ZINK_PIPELINE_MESH]) {
      ctx->pipeline_changed[ZINK_PIPELINE_MESH] = true;
      zink_select_draw_mesh_tasks(ctx);
   }
   /* flush after 30000 draws and dispatches */
   if (unlikely(work_count >= 30000 || ctx->oom_flush) && !ctx->unordered_blitting)
      pctx->flush(pctx, NULL, 0);
}

/* the draws which have dirty atoms: out of line, so the draws without any stay compact */
template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED, zink_draw_vbs VBS>
static void ATTRIBUTE_NOINLINE
zink_draw_atoms(struct zink_context *ctx,
                const struct pipe_draw_info *dinfo,
                unsigned drawid_offset,
                const struct pipe_draw_indirect_info *dindirect,
                const struct pipe_draw_start_count_bias *draws,
                unsigned num_draws,
                struct pipe_vertex_state *vstate,
                uint32_t partial_velem_mask,
                struct zink_so_target *so_target,
                struct pipe_resource *index_buffer,
                unsigned index_offset,
                unsigned index_size,
                bool need_index_buffer_unref,
                unsigned work_count,
                uint32_t skip_atoms)
{
   struct pipe_context *pctx = &ctx->base;
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_batch_state *bs = ctx->bs;
   enum mesa_prim mode = (enum mesa_prim)dinfo->mode;
   const bool dynamic_vertex_input = DYNAMIC_STATE == ZINK_DYNAMIC_VERTEX_INPUT ||
                                     DYNAMIC_STATE == ZINK_DYNAMIC_VERTEX_INPUT2;
   /* these are set by the atoms */
   bool have_streamout = false;
   bool needs_drawid = false;
   bool pipeline_changed = false;

   if (ctx->dirty_atoms & ~skip_atoms) {
      /* inside a renderpass, the renderpass info only has to be parsed again after a new fs or dsa state was bound */
      if (ctx->rp_tc_info_updated &&
          (ctx->dirty_atoms & (BITFIELD_BIT(ZINK_ATOM_PIPELINE) | BITFIELD_BIT(ZINK_ATOM_DSA))))
         zink_mark_atom_dirty(ctx, ZINK_ATOM_RENDERPASS);

      /* an atom can dirty the ones after it, so the mask is read again after every atom */
      uint32_t pending = ZINK_DRAW_ATOMS & ~skip_atoms;
      uint32_t atoms;
      while ((atoms = ctx->dirty_atoms & pending)) {
         const enum zink_atom atom = (enum zink_atom)u_bit_scan(&atoms);
         pending &= ~BITFIELD_MASK(atom + 1);
         ctx->dirty_atoms &= ~BITFIELD_BIT(atom);
         switch (atom) {
         case ZINK_ATOM_MEMORY_BARRIER:
            if (ctx->memory_barrier) {
               if (ctx->blitting)
                  zink_mark_atom_dirty(ctx, ZINK_ATOM_MEMORY_BARRIER);
               else
                  zink_flush_memory_barrier(ctx);
            }
            break;
         case ZINK_ATOM_STREAMOUT:
            if (!ctx->num_so_targets)
               break;
            zink_mark_atom_dirty(ctx, ZINK_ATOM_STREAMOUT);
            have_streamout = true;
            zink_emit_xfb_counter_barrier(ctx);
            if (ctx->dirty_so_targets) {
               /* have to loop here and below because barriers must be emitted out of renderpass,
                * but xfb buffers can't be bound before the renderpass is active to avoid
                * breaking from recursion
                */
               for (unsigned i = 0; i < ctx->num_so_targets; i++) {
                  struct zink_so_target *t = (struct zink_so_target *)ctx->so_targets[i];
                  if (t) {
                     struct zink_resource *res = zink_resource(t->base.buffer);
                     zink_screen(ctx->base.screen)->buffer_barrier(ctx, res,
                                                  VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT);
                     if (!ctx->unordered_blitting)
                        zink_resource_disable_unordered(res, true);
                  }
               }
            }
            break;
         case ZINK_ATOM_BARRIERS:
            /* this may re-emit draw buffer barriers, but such synchronization is harmless */
            if (!ctx->blitting)
               zink_update_barriers(ctx, false, index_buffer, dindirect ? dindirect->buffer : NULL, dindirect ? dindirect->indirect_draw_count : NULL);
            else if (ctx->need_barriers[0]->entries)
               zink_mark_atom_dirty(ctx, ZINK_ATOM_BARRIERS);
            break;
         case ZINK_ATOM_GS_QUERIES:
            if (ctx->vertices_query || !list_is_empty(&ctx->primitives_generated_queries)) {
               zink_mark_atom_dirty(ctx, ZINK_ATOM_GS_QUERIES);
               zink_query_update_gs_states(ctx);
            }
            break;
         case ZINK_ATOM_DEBUG_SYNC:
            if (unlikely(zink_debug & ZINK_DEBUG_SYNC)) {
               zink_mark_atom_dirty(ctx, ZINK_ATOM_DEBUG_SYNC);
               zink_batch_no_rp(ctx);
               VkMemoryBarrier mb;
               mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
               mb.pNext = NULL;
               mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
               mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
               VKSCR(CmdPipelineBarrier)(ctx->bs->cmdbuf,
                                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                         0, 1, &mb, 0, NULL, 0, NULL);
            }
            break;
         case ZINK_ATOM_RENDERPASS:
            /* starting the renderpass can draw, and that draw has to start a renderpass of
             * its own: keep the atom dirty until this one has started
             */
            zink_mark_atom_dirty(ctx, ZINK_ATOM_RENDERPASS);
            zink_batch_rp(ctx);
            /* check dead swapchain */
            if (unlikely(!ctx->in_rp)) {
               if (VBS == ZINK_VBS_UNOWNED)
                  zink_mark_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS);
               return;
            }
            zink_set_atom_dirty(ctx, ZINK_ATOM_RENDERPASS, ctx->rp_layout_changed);
            if (BATCH_CHANGED) {
               zink_update_descriptor_refs(ctx, false);
               /* starting the renderpass can draw, and that draw has emitted its own state */
               ctx->dirty_atoms |= ZINK_DRAW_ATOMS & ~BITFIELD_MASK(ZINK_ATOM_RENDERPASS + 1);
            }
            mark_draw_atoms<DYNAMIC_STATE>(ctx, dinfo);
            break;
         case ZINK_ATOM_DRAW_ID: {
            bool drawid_broken = false;
            if (ctx->shader_reads_drawid) {
               zink_mark_atom_dirty(ctx, ZINK_ATOM_DRAW_ID);
               if (!dindirect || !dindirect->buffer)
                  drawid_broken = (drawid_offset != 0 ||
                                  (!HAS_MULTIDRAW && num_draws > 1) ||
                                  (HAS_MULTIDRAW && num_draws > 1 && !dinfo->increment_draw_id));
            }
            if (drawid_broken != zink_get_last_vertex_key(ctx)->push_drawid)
               zink_set_last_vertex_key(ctx)->push_drawid = drawid_broken;
            needs_drawid = drawid_broken;
            break;
         }
         case ZINK_ATOM_PRIMITIVE: {
            const struct zink_rasterizer_state *rast_state = ctx->rast_state;
            enum mesa_prim rast_prim = zink_rast_prim(ctx, dinfo);
            if (rast_prim != ctx->gfx_pipeline_state.rast_prim) {
               bool points_changed =
                  (ctx->gfx_pipeline_state.rast_prim == MESA_PRIM_POINTS) !=
                  (rast_prim == MESA_PRIM_POINTS);

               static bool rect_warned = false;
               if (DYNAMIC_STATE >= ZINK_DYNAMIC_STATE3 && rast_prim == MESA_PRIM_LINES && !rect_warned &&
                   (VkLineRasterizationMode)rast_state->hw_state.line_mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR) {
                  if (screen->info.line_rast_feats.rectangularLines)
                     rect_warned = true;
                  else
                     warn_missing_feature(rect_warned, "rectangularLines");
               }

               ctx->gfx_pipeline_state.rast_prim = rast_prim;
               zink_mark_atom_dirty(ctx, ZINK_ATOM_PIPELINE);
               if (rast_prim == MESA_PRIM_LINES)
                  zink_mark_atom_dirty(ctx, ZINK_ATOM_LINE_WIDTH);

               if (points_changed && rast_state->base.point_quad_rasterization)
                  zink_set_fs_point_coord_key(ctx);
            }
            const bool mode_changed = ctx->gfx_pipeline_state.gfx_prim_mode != mode;
            if (DYNAMIC_STATE != ZINK_NO_DYNAMIC_STATE && (BATCH_CHANGED || mode_changed))
               VKCTX(CmdSetPrimitiveTopology)(bs->cmdbuf, zink_primitive_topology(mode));
            /* a pipeline only draws its own topology, or its own topology class with dynamic topology */
            if (mode_changed && get_gfx_pipeline_idx<DYNAMIC_STATE>(screen, mode) != ctx->gfx_pipeline_state.idx)
               zink_mark_atom_dirty(ctx, ZINK_ATOM_PIPELINE);
            ctx->gfx_pipeline_state.gfx_prim_mode = mode;
            /* with optimal keys, only these can need a generated geometry shader or have one to unbind */
            if (!screen->optimal_keys || ctx->is_generated_gs_bound || mode == MESA_PRIM_QUADS ||
                zink_get_gs_key(ctx)->lower_gl_point ||
                (ctx->gfx_stages[MESA_SHADER_VERTEX] && ctx->gfx_stages[MESA_SHADER_VERTEX]->has_edgeflags))
               zink_set_primitive_emulation_keys(ctx);

            if (DYNAMIC_STATE >= ZINK_DYNAMIC_STATE2) {
               if (BATCH_CHANGED || ctx->primitive_restart != dinfo->primitive_restart) {
                  VKCTX(CmdSetPrimitiveRestartEnable)(bs->cmdbuf, dinfo->primitive_restart);
                  ctx->primitive_restart = dinfo->primitive_restart;
               }
            } else if (ctx->gfx_pipeline_state.dyn_state2.primitive_restart != dinfo->primitive_restart) {
               ctx->gfx_pipeline_state.dyn_state2.primitive_restart = dinfo->primitive_restart;
               zink_gfx_pipeline_state_dirty(ctx);
            }

            /* a psiz output injected by the frontend is only needed when points are rasterized */
            const bool strip_psiz = ctx->last_vertex_stage->non_fs.has_injected_psiz &&
                                    ctx->gfx_pipeline_state.rast_prim != MESA_PRIM_POINTS;
            if (strip_psiz != zink_get_last_vertex_key(ctx)->strip_psiz)
               zink_set_last_vertex_key(ctx)->strip_psiz = strip_psiz;

            /* a generated geometry shader may turn lines and points into triangles */
            bool using_depth_bias = zink_uses_depth_bias(rast_state, zink_prim_type(ctx, dinfo),
                                                         !(ctx->last_vertex_stage && ctx->last_vertex_stage->non_fs.is_generated));
            if (using_depth_bias != ctx->was_using_depth_bias) {
               ctx->was_using_depth_bias = using_depth_bias;
               zink_mark_atom_dirty(ctx, ZINK_ATOM_DEPTH_BIAS);
            }
            break;
         }
         case ZINK_ATOM_PIPELINE:
            if (!update_gfx_pipeline<DYNAMIC_STATE, BATCH_CHANGED>(ctx, bs, mode, &pipeline_changed)) {
               zink_mark_atom_dirty(ctx, ZINK_ATOM_PIPELINE);
               if (VBS == ZINK_VBS_UNOWNED)
                  zink_mark_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS);
               return;
            }
            /* the descriptor sets are bound for the pipeline's layout */
            if (pipeline_changed) {
               zink_mark_atom_dirty(ctx, ZINK_ATOM_DESCRIPTORS);
               if (ctx->di.any_bindless_dirty)
                  zink_mark_atom_dirty(ctx, ZINK_ATOM_BINDLESS);
            }
            zink_set_atom_dirty(ctx, ZINK_ATOM_PUSH_CONSTANTS,
                                ctx->shader_reads_basevertex || !screen->optimal_keys ||
                                (ctx->curr_program->shaders[MESA_SHADER_TESS_CTRL] &&
                                 ctx->curr_program->shaders[MESA_SHADER_TESS_CTRL]->non_fs.is_generated));
            break;
         case ZINK_ATOM_VERTEX_BUFFERS:
            if (VBS == ZINK_VBS_BOUND)
               bind_vertex_buffers<DYNAMIC_STATE>(ctx, ctx->vertex_buffers);
            break;
         case ZINK_ATOM_VERTEX_INPUT:
            if (VBS != ZINK_VBS_VERTEX_STATE && dynamic_vertex_input)
               VKCTX(CmdSetVertexInputEXT)(ctx->bs->cmdbuf,
                                           ctx->element_state->hw_state.num_bindings, ctx->element_state->hw_state.dynbindings,
                                           ctx->element_state->hw_state.num_attribs, ctx->element_state->hw_state.dynattribs);
            break;
         case ZINK_ATOM_DESCRIPTORS:
            if (zink_program_has_descriptors(&ctx->curr_program->base))
               zink_descriptors_update(ctx, ZINK_PIPELINE_GFX);
            break;
         case ZINK_ATOM_BINDLESS:
            if (ctx->di.any_bindless_dirty &&
                /* some apps (d3dretrace) call MakeTextureHandleResidentARB randomly */
                zink_program_has_descriptors(&ctx->curr_program->base) &&
                ctx->curr_program->base.dd.bindless)
               zink_descriptors_update_bindless(ctx);
            break;
         case ZINK_ATOM_PUSH_CONSTANTS:
            /* ZINK_ATOM_PIPELINE unsets this */
            zink_mark_atom_dirty(ctx, ZINK_ATOM_PUSH_CONSTANTS);
            if (ctx->shader_reads_basevertex) {
               unsigned draw_mode_is_indexed = index_size > 0;
               VKCTX(CmdPushConstants)(bs->cmdbuf, ctx->curr_program->base.layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 offsetof(struct zink_gfx_push_constant, draw_mode_is_indexed), sizeof(unsigned),
                                 &draw_mode_is_indexed);
            }
            if (ctx->curr_program->shaders[MESA_SHADER_TESS_CTRL] &&
                ctx->curr_program->shaders[MESA_SHADER_TESS_CTRL]->non_fs.is_generated) {
               VKCTX(CmdPushConstants)(bs->cmdbuf, ctx->curr_program->base.layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 offsetof(struct zink_gfx_push_constant, default_inner_level), sizeof(float) * 6,
                                 &ctx->tess_levels[0]);
            }

            if (!screen->optimal_keys) {
               if (zink_get_fs_key(ctx)->lower_line_stipple ||
                   zink_get_gs_key(ctx)->lower_gl_point ||
                   zink_get_fs_key(ctx)->lower_line_smooth) {

                  assert(zink_get_gs_key(ctx)->lower_line_stipple ==
                         zink_get_fs_key(ctx)->lower_line_stipple);

                  assert(zink_get_gs_key(ctx)->lower_line_smooth ==
                         zink_get_fs_key(ctx)->lower_line_smooth);

                  float viewport_scale[2] = {
                     ctx->vp_state.viewport_states[0].scale[0],
                     ctx->vp_state.viewport_states[0].scale[1]
                  };
                  VKCTX(CmdPushConstants)(bs->cmdbuf,
                                          ctx->curr_program->base.layout,
                                          VK_SHADER_STAGE_ALL_GRAPHICS,
                                          offsetof(struct zink_gfx_push_constant, viewport_scale),
                                          sizeof(float) * 2, &viewport_scale);

                  uint32_t stipple = ctx->rast_state->base.line_stipple_pattern;
                  stipple |= ctx->rast_state->base.line_stipple_factor << 16;
                  VKCTX(CmdPushConstants)(bs->cmdbuf,
                                          ctx->curr_program->base.layout,
                                          VK_SHADER_STAGE_ALL_GRAPHICS,
                                          offsetof(struct zink_gfx_push_constant, line_stipple_pattern),
                                          sizeof(uint32_t), &stipple);

                  if (ctx->gfx_pipeline_state.shader_keys.key[MESA_SHADER_FRAGMENT].key.fs.lower_line_smooth) {
                     float line_width = ctx->rast_state->base.line_width;
                     VKCTX(CmdPushConstants)(bs->cmdbuf,
                                             ctx->curr_program->base.layout,
                                             VK_SHADER_STAGE_ALL_GRAPHICS,
                                             offsetof(struct zink_gfx_push_constant, line_width),
                                             sizeof(uint32_t), &line_width);
                  }
               }
            }
            break;
         default:
            UNREACHABLE("not a draw atom");
         }
      }

      /* the line width is only set while lines are drawn: ZINK_ATOM_PRIMITIVE dirties it when that starts */
      if (!BATCH_CHANGED && ctx->gfx_pipeline_state.rast_prim != MESA_PRIM_LINES)
         zink_set_atom_dirty(ctx, ZINK_ATOM_LINE_WIDTH, false);
   }
   emit_dynamic_state<DYNAMIC_STATE, BATCH_CHANGED>(ctx, pipeline_changed, ctx->vp_state.num_viewports, ctx->curr_program->base.uses_shobj, 0);
   /* this draw replaces the vertex input and buffers a vertex-state draw left bound */
   if (VBS != ZINK_VBS_VERTEX_STATE)
      ctx->vstate_bound.input_valid = false;

   if (BATCH_CHANGED) {
      ctx->pipeline_changed[ZINK_PIPELINE_GFX] = false;
      zink_select_draw_vbo(ctx);
   }

   zink_draw_finish<HAS_MULTIDRAW, DYNAMIC_STATE, VBS, false>(ctx, bs, dinfo, drawid_offset, dindirect, draws, num_draws, so_target,
                                                              index_buffer, index_offset, index_size, need_index_buffer_unref,
                                                              work_count, have_streamout, needs_drawid);
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED, zink_draw_vbs VBS>
ALWAYS_INLINE static void
zink_draw(struct pipe_context *pctx,
          const struct pipe_draw_info *dinfo,
          unsigned drawid_offset,
          const struct pipe_draw_indirect_info *dindirect,
          const struct pipe_draw_start_count_bias *draws,
          unsigned num_draws,
          struct pipe_vertex_state *vstate,
          uint32_t partial_velem_mask)
{
   MESA_TRACE_FUNC();
   if (!dindirect && (!draws[0].count || !dinfo->instance_count)) {
      if (VBS == ZINK_VBS_UNOWNED)
         zink_mark_atom_dirty(zink_context(pctx), ZINK_ATOM_VERTEX_BUFFERS);
      return;
   }

   struct zink_context *ctx = zink_context(pctx);
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_batch_state *bs = ctx->bs;
   struct zink_so_target *so_target =
      dindirect && dindirect->count_from_stream_output ?
         zink_so_target(dindirect->count_from_stream_output) : NULL;
   bool need_index_buffer_unref = false;
   unsigned work_count = ctx->work_count;

   ctx->rp_draw = true;

   /* a new cmdbuf has none of the state */
   if (BATCH_CHANGED)
      ctx->dirty_atoms |= ZINK_DRAW_ATOMS;

   /* MSVC resolves "struct zink_context" inside offsetof() to the
    * zink_context() cast function, so these go through typedefs
    */
   typedef struct zink_context zink_context_type;
   typedef struct zink_screen zink_screen_type;
   STATIC_ASSERT(offsetof(zink_context_type, image_rebind_counter) ==
                 offsetof(zink_context_type, buffer_rebind_counter) + sizeof(unsigned));
   STATIC_ASSERT(offsetof(zink_screen_type, image_rebind_counter) ==
                 offsetof(zink_screen_type, buffer_rebind_counter) + sizeof(unsigned));
   /* both counters at once: they only differ after something was rebound */
   uint64_t ctx_rebinds, screen_rebinds;
   memcpy(&ctx_rebinds, &ctx->buffer_rebind_counter, sizeof(ctx_rebinds));
   memcpy(&screen_rebinds, &screen->buffer_rebind_counter, sizeof(screen_rebinds));
   if (unlikely(ctx_rebinds != screen_rebinds)) {
      if (ctx->buffer_rebind_counter < screen->buffer_rebind_counter && !ctx->blitting) {
         ctx->buffer_rebind_counter = screen->buffer_rebind_counter;
         zink_rebind_all_buffers(ctx);
      }

      if (ctx->image_rebind_counter < screen->image_rebind_counter && !ctx->blitting) {
         ctx->image_rebind_counter = screen->image_rebind_counter;
         zink_rebind_all_images(ctx);
      }
   }

   /* the buffers which come with the draw are synchronized here: this can end the renderpass */
   unsigned index_offset = 0;
   unsigned index_size = dinfo->index_size;
   struct pipe_resource *index_buffer = NULL;
   if (index_size > 0) {
      if (dinfo->has_user_indices) {
         if (!util_upload_index_buffer(pctx, dinfo, &draws[0], &index_buffer, &index_offset, 4)) {
            debug_printf("util_upload_index_buffer() failed\n");
            if (VBS == ZINK_VBS_UNOWNED)
               zink_mark_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS);
            return;
         }
         /* this will have extra refs from tc */
         if (screen->threaded)
            zink_batch_reference_resource_move(ctx, zink_resource(index_buffer));
         else
            zink_batch_reference_resource(ctx, zink_resource(index_buffer));
      } else if (BATCH_CHANGED || ctx->index_buffer != dinfo->index.resource || ctx->index_size != dinfo->index_size) {
         ctx->index_buffer = index_buffer = dinfo->index.resource;
         ctx->index_size = dinfo->index_size;
         zink_batch_resource_usage_set(ctx->bs, zink_resource(index_buffer), false, true);
      }
      assert(index_size <= 4 && index_size != 3);
      assert(index_size != 1 || screen->info.have_EXT_index_type_uint8);
      if (index_buffer)
         check_buffer_barrier(ctx, index_buffer, VK_ACCESS_INDEX_READ_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
   } else {
      ctx->index_buffer = NULL;
   }
   /* zink_draw_vbo_buffers has synchronized its buffers already */
   if (VBS == ZINK_VBS_BOUND && ctx->vertex_buffers_unowned && zink_is_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS))
      barrier_unowned_vertex_buffers<false>(ctx, ctx->vertex_buffers, ctx->vertex_buffers_count);
   if (unlikely(dindirect)) {
      if (dindirect->buffer) {
         check_buffer_barrier(ctx, dindirect->buffer,
                              VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
         if (dindirect->indirect_draw_count)
            check_buffer_barrier(ctx, dindirect->indirect_draw_count,
                                 VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
      }
      /* ensure synchronization between doing streamout with counter buffer
       * and using counter buffer for indirect draw
       */
      if (so_target && so_target->counter_buffer_valid) {
         struct zink_resource *res = zink_resource(so_target->counter_buffer);
         zink_screen(ctx->base.screen)->buffer_barrier(ctx, res,
                                      VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
                                      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
         if (!ctx->unordered_blitting)
            zink_resource_disable_unordered(res, false);
      }
   }

   ctx->was_line_loop = dinfo->was_line_loop;
   mark_draw_atoms<DYNAMIC_STATE>(ctx, dinfo);
   validate_atoms(ctx);

   /* the buffers of draw_vbo_buffers change with every draw, so they are bound after the atoms;
    * vertex-state draws bind their own buffers and vertex input, and leave both atoms dirty for
    * the next draw which doesn't
    */
   const uint32_t skip_atoms = VBS == ZINK_VBS_UNOWNED ? BITFIELD_BIT(ZINK_ATOM_VERTEX_BUFFERS) :
                               VBS == ZINK_VBS_VERTEX_STATE ? BITFIELD_BIT(ZINK_ATOM_VERTEX_BUFFERS) |
                                                              BITFIELD_BIT(ZINK_ATOM_VERTEX_INPUT) :
                               0;
   /* nothing has changed since the last draw except the buffers which come with this one */
   if (!BATCH_CHANGED && !(ctx->dirty_atoms & ~skip_atoms) && !dindirect &&
       !(ctx->depth_op_promoted && !ctx->can_promote_depth_op) && !zink_tracing) {
      zink_draw_finish<HAS_MULTIDRAW, DYNAMIC_STATE, VBS, true>(ctx, bs, dinfo, drawid_offset, NULL, draws, num_draws, NULL,
                                                               index_buffer, index_offset, index_size, need_index_buffer_unref,
                                                               work_count, false, false);
      return;
   }
   zink_draw_atoms<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, VBS>(ctx, dinfo, drawid_offset, dindirect, draws, num_draws, vstate, partial_velem_mask,
                                                                    so_target, index_buffer, index_offset, index_size,
                                                                    need_index_buffer_unref, work_count, skip_atoms);
}


template <bool BATCH_CHANGED>
static bool
update_mesh_pipeline(struct zink_context *ctx, struct zink_batch_state *bs, bool *pipeline_changed_out)
{
   VkPipeline prev_pipeline = ctx->gfx_pipeline_state.pipeline;
   const struct zink_screen *screen = zink_screen(ctx->base.screen);
   if (BATCH_CHANGED && ctx->mesh_program) {
      if (zink_gfx_program_needs_replacement(ctx->mesh_program))
         zink_gfx_program_queue_replacement(ctx, ctx->mesh_program);
      else if (zink_gfx_program_has_replacement(ctx->mesh_program))
         ctx->mesh_dirty = true;
   }
   bool shaders_changed = ctx->mesh_dirty || ctx->dirty_mesh_stages;
   zink_mesh_program_update_optimal(ctx);
   bool pipeline_changed = false;
   VkPipeline pipeline = VK_NULL_HANDLE;
   if (!ctx->mesh_program->base.uses_shobj) {
      if (screen->info.have_EXT_graphics_pipeline_library)
         pipeline = zink_get_gfx_pipeline<ZINK_DYNAMIC_STATE3, true, true>(ctx, ctx->mesh_program, &ctx->gfx_pipeline_state, MESA_PRIM_COUNT);
      else
         pipeline = zink_get_gfx_pipeline<ZINK_DYNAMIC_STATE3, false, true>(ctx, ctx->mesh_program, &ctx->gfx_pipeline_state, MESA_PRIM_COUNT);
      /* creation failed and was logged there; the caller skips the draw */
      if (unlikely(pipeline == VK_NULL_HANDLE))
         return false;
      pipeline_changed = prev_pipeline != pipeline || ctx->shobj_draw;
      if (BATCH_CHANGED || pipeline_changed)
         VKCTX(CmdBindPipeline)(bs->cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      ctx->shobj_draw = false;
   } else {
      if (BATCH_CHANGED || shaders_changed || !ctx->shobj_draw) {
         VkShaderStageFlagBits stages[] = {
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM,
            VK_SHADER_STAGE_TASK_BIT_EXT,
            VK_SHADER_STAGE_MESH_BIT_EXT,
         };
         /* always rebind all stages */
         VKCTX(CmdBindShadersEXT)(bs->cmdbuf, ZINK_GFX_SHADER_COUNT, stages, ctx->mesh_program->objects);
         VKCTX(CmdBindShadersEXT)(bs->cmdbuf, 2, &stages[MESA_SHADER_TASK], &ctx->mesh_program->objects[MESA_SHADER_TASK]);
         VKCTX(CmdSetDepthBiasEnable)(bs->cmdbuf, VK_TRUE);
         pipeline_changed = true;
      }
      ctx->shobj_draw = true;
   }
   *pipeline_changed_out = pipeline_changed;
   return true;
}

template <bool BATCH_CHANGED>
void
zink_draw_mesh_tasks(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
   MESA_TRACE_FUNC();

   struct zink_context *ctx = zink_context(pctx);
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_batch_state *bs = ctx->bs;
   unsigned work_count = ctx->work_count;

   ctx->rp_draw = true;

   if (ctx->memory_barrier && !ctx->blitting)
      zink_flush_memory_barrier(ctx);

   if (unlikely(ctx->buffer_rebind_counter < screen->buffer_rebind_counter && !ctx->blitting)) {
      ctx->buffer_rebind_counter = screen->buffer_rebind_counter;
      zink_rebind_all_buffers(ctx);
   }

   if (unlikely(ctx->image_rebind_counter < screen->image_rebind_counter && !ctx->blitting)) {
      ctx->image_rebind_counter = screen->image_rebind_counter;
      zink_rebind_all_images(ctx);
   }

   if (info->indirect) {
      check_buffer_barrier(ctx, info->indirect,
                           VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
      if (info->indirect_draw_count)
         check_buffer_barrier(ctx, info->indirect_draw_count,
                              VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
   }
   zink_update_barriers(ctx, false, NULL, info->indirect, info->indirect_draw_count);

   if (unlikely(zink_debug & ZINK_DEBUG_SYNC)) {
      zink_batch_no_rp(ctx);
      VkMemoryBarrier mb;
      mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      mb.pNext = NULL;
      mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      VKSCR(CmdPipelineBarrier)(ctx->bs->cmdbuf,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, 1, &mb, 0, NULL, 0, NULL);
   }

   bool need_rp_update = !ctx->in_rp || ctx->rp_changed;
   zink_batch_rp(ctx);
   /* check dead swapchain */
   if (unlikely(!ctx->in_rp))
      return;

   if (BATCH_CHANGED)
      zink_update_descriptor_refs(ctx, false);

   bool pipeline_changed = false;
   if (need_rp_update || ctx->mesh_dirty || ctx->dirty_mesh_stages || BATCH_CHANGED) {
      if (!update_mesh_pipeline<BATCH_CHANGED>(ctx, bs, &pipeline_changed))
         return;
   }

   bool using_depth_bias = zink_uses_depth_bias(ctx->rast_state, ctx->gfx_stages[MESA_SHADER_MESH]->info.mesh.primitive_type, true);
   if (using_depth_bias != ctx->was_using_depth_bias) {
      ctx->was_using_depth_bias = using_depth_bias;
      zink_mark_atom_dirty(ctx, ZINK_ATOM_DEPTH_BIAS);
      /* the next zink_draw decides this from its own primitive */
      zink_mark_atom_dirty(ctx, ZINK_ATOM_PRIMITIVE);
   }
   /* the line width is only set by zink_draw */
   emit_dynamic_state<ZINK_DYNAMIC_STATE3, BATCH_CHANGED>(ctx, pipeline_changed, ctx->vp_state.mesh_num_viewports, ctx->mesh_program->base.uses_shobj,
                                                          BITFIELD_BIT(ZINK_ATOM_LINE_WIDTH));

   if (zink_program_has_descriptors(&ctx->mesh_program->base) && (BATCH_CHANGED || ctx->dd.push_state_changed[ZINK_PIPELINE_MESH] || ctx->dd.state_changed[ZINK_PIPELINE_MESH] || pipeline_changed))
      zink_descriptors_update(ctx, ZINK_PIPELINE_MESH);

   if (ctx->di.any_bindless_dirty &&
       /* some apps (d3dretrace) call MakeTextureHandleResidentARB randomly */
       zink_program_has_descriptors(&ctx->mesh_program->base) &&
       ctx->mesh_program->base.dd.bindless)
      zink_descriptors_update_bindless(ctx);

   if (BATCH_CHANGED) {
      ctx->pipeline_changed[ZINK_PIPELINE_MESH] = false;
      zink_select_draw_mesh_tasks(ctx);
   }

   bool marker = false;
   if (unlikely(zink_tracing)) {
      marker = zink_cmd_debug_marker_begin(ctx, VK_NULL_HANDLE, "draw_mesh(%u cbufs|%s, %dx%d)",
                                             ctx->fb_state.nr_cbufs,
                                             ctx->fb_state.zsbuf.texture ? "zsbuf" : "",
                                             lround(ctx->fb_state.width), lround(ctx->fb_state.height));
   }

   if (info->indirect) {
      struct zink_resource *indirect = zink_resource(info->indirect);
      zink_batch_reference_resource_rw(ctx, indirect, false);
      if (info->indirect_draw_count) {
         struct zink_resource *indirect_draw_count = zink_resource(info->indirect_draw_count);
         zink_batch_reference_resource_rw(ctx, indirect_draw_count, false);
         VKCTX(CmdDrawMeshTasksIndirectCountEXT)(bs->cmdbuf, indirect->obj->buffer, info->indirect_offset,
                                                 indirect_draw_count->obj->buffer, info->indirect_draw_count_offset,
                                                 info->draw_count, info->indirect_stride);
      } else
         VKCTX(CmdDrawMeshTasksIndirectEXT)(bs->cmdbuf, indirect->obj->buffer, info->indirect_offset, info->draw_count, info->indirect_stride);
   } else {
      VKCTX(CmdDrawMeshTasksEXT)(bs->cmdbuf, info->grid[0], info->grid[1], info->grid[2]);
   }

   if (unlikely(zink_tracing))
      zink_cmd_debug_marker_end(ctx, bs->cmdbuf, marker);

   ctx->last_work_was_compute = false;
   ctx->work_count++;
   if (!ctx->pipeline_changed[ZINK_PIPELINE_GFX]) {
      ctx->pipeline_changed[ZINK_PIPELINE_GFX] = true;
      zink_select_draw_vbo(ctx);
   }
   /* flush after 30000 draws and dispatches */
   if (!ctx->unordered_blitting && (unlikely(work_count >= 30000) || ctx->oom_flush))
      pctx->flush(pctx, NULL, 0);
}

/* NVIDIA and NVK are slow for list topologies with restart indices: draw a compacted copy instead */
static ALWAYS_INLINE bool
zink_list_restart_compact(struct zink_context *ctx, const struct pipe_draw_info *dinfo,
                          unsigned drawid_offset, const struct pipe_draw_indirect_info *indirect,
                          const struct pipe_draw_start_count_bias *draws, unsigned num_draws,
                          struct pipe_draw_info *info_out, struct pipe_draw_indirect_info *indirect_out,
                          struct pipe_draw_start_count_bias *draw_out)
{
   const struct zink_screen *screen = zink_screen(ctx->base.screen);
   return screen->driver_workarounds.compact_list_restart && !indirect && !drawid_offset &&
          dinfo->index_size && !dinfo->has_user_indices && num_draws == 1 && dinfo->instance_count &&
          !ctx->blitting && !ctx->unordered_blitting && !ctx->num_so_targets &&
          (!dinfo->start_instance || screen->info.feats.features.drawIndirectFirstInstance) &&
          zink_compact_list_restart(ctx, dinfo, &draws[0], info_out, indirect_out, draw_out);
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED>
static void
zink_draw_vbo(struct pipe_context *pctx,
              const struct pipe_draw_info *info,
              unsigned drawid_offset,
              const struct pipe_draw_indirect_info *indirect,
              const struct pipe_draw_start_count_bias *draws,
              unsigned num_draws)
{
   if (unlikely(info->primitive_restart)) {
      struct zink_context *ctx = zink_context(pctx);
      struct pipe_draw_info compact_info;
      struct pipe_draw_indirect_info compact_indirect;
      struct pipe_draw_start_count_bias compact_draw;
      if (zink_list_restart_compact(ctx, info, drawid_offset, indirect, draws, num_draws, &compact_info, &compact_indirect, &compact_draw)) {
         zink_draw<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, ZINK_VBS_BOUND>(pctx, &compact_info, drawid_offset, &compact_indirect,
                                                                               &compact_draw, 1, NULL, 0);
         return;
      }
   }
   zink_draw<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, ZINK_VBS_BOUND>(pctx, info, drawid_offset, indirect, draws, num_draws, NULL, 0);
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED>
static void
zink_draw_vbo_buffers(struct pipe_context *pctx,
                      const struct pipe_draw_info *dinfo,
                      const struct pipe_vertex_buffer *buffers,
                      unsigned buffer_count,
                      const struct pipe_draw_start_count_bias *draws,
                      unsigned num_draws)
{
   MESA_TRACE_FUNC();

   struct zink_context *ctx = zink_context(pctx);

   struct pipe_draw_info compact_info;
   struct pipe_draw_indirect_info compact_indirect;
   struct pipe_draw_start_count_bias compact_draw;
   /* this goes before the buffers are synchronized, since the pass can end the batch */
   const bool compacted = unlikely(dinfo->primitive_restart) &&
                          zink_list_restart_compact(ctx, dinfo, 0, NULL, draws, num_draws, &compact_info, &compact_indirect,
                                                    &compact_draw);

   if (ctx->in_rp)
      ctx->can_promote_depth_op = false;

   if (unlikely(!ctx->vertex_buffers_unowned || buffer_count != ctx->vertex_buffers_count))
      zink_set_vertex_buffers_unowned(ctx, buffer_count, buffers);
   /* these are used until the next set_vertex_buffers: zink_draw binds them, or dirties
    * ZINK_ATOM_VERTEX_BUFFERS when it returns without drawing
    */
   barrier_unowned_vertex_buffers<true>(ctx, buffers, buffer_count);
   if (unlikely(compacted)) {
      zink_draw<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, ZINK_VBS_UNOWNED>(pctx, &compact_info, 0, &compact_indirect,
                                                                             &compact_draw, 1, NULL, 0);
      return;
   }
   zink_draw<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, ZINK_VBS_UNOWNED>(pctx, dinfo, 0, NULL, draws, num_draws, NULL, 0);
}

template <util_popcnt HAS_POPCNT>
static void
zink_vertex_state_mask(struct zink_context *ctx, struct pipe_vertex_state *vstate, uint32_t partial_velem_mask)
{
   struct zink_vertex_state *zstate = (struct zink_vertex_state *)vstate;
   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;

   if (partial_velem_mask == vstate->input.full_velem_mask) {
      VKCTX(CmdSetVertexInputEXT)(cmdbuf,
                                 zstate->velems.hw_state.num_bindings, zstate->velems.hw_state.dynbindings,
                                 zstate->velems.hw_state.num_attribs, zstate->velems.hw_state.dynattribs);
      return;
   }

   VkVertexInputAttributeDescription2EXT dynattribs[PIPE_MAX_ATTRIBS];
   unsigned num_attribs = 0;
   u_foreach_bit(elem, vstate->input.full_velem_mask & partial_velem_mask) {
      unsigned idx = util_bitcount_fast<HAS_POPCNT>(vstate->input.full_velem_mask & BITFIELD_MASK(elem));
      dynattribs[num_attribs] = zstate->velems.hw_state.dynattribs[idx];
      dynattribs[num_attribs].location = num_attribs;
      num_attribs++;
   }

   VKCTX(CmdSetVertexInputEXT)(cmdbuf,
                               zstate->velems.hw_state.num_bindings, zstate->velems.hw_state.dynbindings,
                               num_attribs, dynattribs);
}

template <util_popcnt HAS_POPCNT>
static void
zink_bind_vertex_state(struct zink_context *ctx, struct pipe_vertex_state *vstate, uint32_t partial_velem_mask)
{
   MESA_TRACE_FUNC();
   struct zink_vertex_state *zstate = (struct zink_vertex_state *)vstate;
   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;
   if (!vstate->input.vbuffer.buffer.resource)
      return;

   const unsigned num_bindings = zstate->velems.hw_state.num_bindings;
   const unsigned num_attribs = zstate->velems.hw_state.num_attribs;
   /* different vertex states usually share a layout, so compare contents and not pointers */
   if (!ctx->vstate_bound.input_valid ||
       ctx->vstate_bound.full_velem_mask != vstate->input.full_velem_mask ||
       ctx->vstate_bound.partial_velem_mask != partial_velem_mask ||
       ctx->vstate_bound.num_bindings != num_bindings ||
       ctx->vstate_bound.num_attribs != num_attribs ||
       memcmp(ctx->vstate_bound.bindings, zstate->velems.hw_state.dynbindings, num_bindings * sizeof(ctx->vstate_bound.bindings[0])) ||
       memcmp(ctx->vstate_bound.attribs, zstate->velems.hw_state.dynattribs, num_attribs * sizeof(ctx->vstate_bound.attribs[0]))) {
      zink_vertex_state_mask<HAS_POPCNT>(ctx, vstate, partial_velem_mask);
      ctx->vstate_bound.input_valid = true;
      ctx->vstate_bound.full_velem_mask = vstate->input.full_velem_mask;
      ctx->vstate_bound.partial_velem_mask = partial_velem_mask;
      ctx->vstate_bound.num_bindings = num_bindings;
      ctx->vstate_bound.num_attribs = num_attribs;
      memcpy(ctx->vstate_bound.bindings, zstate->velems.hw_state.dynbindings, num_bindings * sizeof(ctx->vstate_bound.bindings[0]));
      memcpy(ctx->vstate_bound.attribs, zstate->velems.hw_state.dynattribs, num_attribs * sizeof(ctx->vstate_bound.attribs[0]));
      ctx->vstate_bound.buffer_valid = false;
   }

   struct zink_resource *res = zink_resource(vstate->input.vbuffer.buffer.resource);
   zink_batch_resource_usage_set(ctx->bs, res, false, true);
   VkDeviceSize offset = vstate->input.vbuffer.buffer_offset;
   if (!ctx->vstate_bound.buffer_valid || ctx->vstate_bound.buffer != res->obj->buffer || ctx->vstate_bound.offset != offset) {
      VKCTX(CmdBindVertexBuffers)(cmdbuf, 0, num_bindings, &res->obj->buffer, &offset);
      ctx->vstate_bound.buffer_valid = true;
      ctx->vstate_bound.buffer = res->obj->buffer;
      ctx->vstate_bound.offset = offset;
   }
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, util_popcnt HAS_POPCNT, bool BATCH_CHANGED>
static void
zink_draw_vertex_state(struct pipe_context *pctx,
                       struct pipe_vertex_state *vstate,
                       uint32_t partial_velem_mask,
                       struct pipe_draw_vertex_state_info info,
                       const struct pipe_draw_start_count_bias *draws,
                       unsigned num_draws)
{
   MESA_TRACE_FUNC();
   struct pipe_draw_info dinfo = {};

   dinfo.mode = info.mode;
   dinfo.index_size = 4;
   dinfo.instance_count = 1;
   dinfo.index.resource = vstate->input.indexbuf;
   struct zink_context *ctx = zink_context(pctx);
   struct zink_resource *res = zink_resource(vstate->input.vbuffer.buffer.resource);
   zink_screen(ctx->base.screen)->buffer_barrier(ctx, res, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
   if (!ctx->unordered_blitting)
      zink_resource_disable_unordered(res, false);
   /* a new command buffer has no vertex input or buffers bound */
   if (BATCH_CHANGED)
      ctx->vstate_bound.input_valid = false;
   zink_bind_vertex_state<HAS_POPCNT>(ctx, vstate, partial_velem_mask);

   zink_draw<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED, ZINK_VBS_VERTEX_STATE>(pctx, &dinfo, 0, NULL, draws, num_draws, vstate, partial_velem_mask);
   /* ensure ctx->vertex_buffers gets rebound on next non-vstate draw */
   zink_mark_atom_dirty(ctx, ZINK_ATOM_VERTEX_BUFFERS);
   /* ensure ctx->element_state gets rebound on next non-vstate draw */
   zink_mark_atom_dirty(ctx, ZINK_ATOM_VERTEX_INPUT);

   if (info.take_vertex_state_ownership)
      pipe_vertex_state_reference(&vstate, NULL);
}

template <bool BATCH_CHANGED>
static void
zink_launch_grid(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
   MESA_TRACE_FUNC();
   struct zink_context *ctx = zink_context(pctx);
   struct zink_batch_state *bs = ctx->bs;
   struct zink_screen *screen = zink_screen(pctx->screen);

   if (unlikely(ctx->buffer_rebind_counter < screen->buffer_rebind_counter)) {
      ctx->buffer_rebind_counter = screen->buffer_rebind_counter;
      zink_rebind_all_buffers(ctx);
   }

   if (unlikely(ctx->image_rebind_counter < screen->image_rebind_counter)) {
      ctx->image_rebind_counter = screen->image_rebind_counter;
      zink_rebind_all_images(ctx);
   }

   if (ctx->render_condition_active)
      zink_start_conditional_render(ctx);

   if (info->indirect) {
      /*
         VK_ACCESS_INDIRECT_COMMAND_READ_BIT specifies read access to indirect command data read as
         part of an indirect build, trace, drawing or dispatching command. Such access occurs in the
         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT pipeline stage.

         - Chapter 7. Synchronization and Cache Control
       */
      check_buffer_barrier(ctx, info->indirect, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
   }

   zink_update_barriers(ctx, true, NULL, info->indirect, NULL);
   if (ctx->memory_barrier)
      zink_flush_memory_barrier(ctx);

   if (unlikely(zink_debug & ZINK_DEBUG_SYNC)) {
      zink_batch_no_rp(ctx);
      VkMemoryBarrier mb;
      mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      mb.pNext = NULL;
      mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      VKSCR(CmdPipelineBarrier)(ctx->bs->cmdbuf,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, 1, &mb, 0, NULL, 0, NULL);
   }

   zink_program_update_compute_pipeline_state(ctx, ctx->curr_compute, info);
   VkPipeline prev_pipeline = ctx->compute_pipeline_state.pipeline;

   if (BATCH_CHANGED) {
      zink_update_descriptor_refs(ctx, true);

      for (unsigned i = 0; i < info->num_globals; i++) {
         struct zink_resource *res = zink_resource(info->globals[i]);
         util_range_add(&res->base.b, &res->valid_buffer_range, 0, res->base.b.width0);
         /* barrier before setting usage so access from completed batches is reset */
         zink_screen(ctx->base.screen)->buffer_barrier(ctx, res, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
         zink_batch_reference_resource_rw(ctx, res, true);
         zink_resource_disable_unordered(res, true);
      }
   }
   if (ctx->compute_dirty) {
      /* update inlinable constants */
      zink_update_compute_program(ctx);
      ctx->compute_dirty = false;
   }

   VkPipeline pipeline = zink_get_compute_pipeline(screen, ctx->curr_compute,
                                               &ctx->compute_pipeline_state);
   /* creation failed and was logged there; skip the dispatch */
   if (unlikely(pipeline == VK_NULL_HANDLE))
      return;

   bool pipeline_changed = prev_pipeline != pipeline;
   if (pipeline_changed || BATCH_CHANGED)
      VKCTX(CmdBindPipeline)(bs->cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   if (BATCH_CHANGED) {
      ctx->pipeline_changed[1] = false;
      zink_select_launch_grid(ctx);
   }

   if (zink_program_has_descriptors(&ctx->curr_compute->base) && (BATCH_CHANGED || ctx->dd.push_state_changed[1] || ctx->dd.state_changed[1] || pipeline_changed))
      zink_descriptors_update(ctx, ZINK_PIPELINE_COMPUTE);
   if (ctx->di.any_bindless_dirty && ctx->curr_compute->base.dd.bindless)
      zink_descriptors_update_bindless(ctx);

   ctx->work_count++;
   zink_batch_no_rp(ctx);
   if (!ctx->queries_disabled)
      zink_resume_cs_query(ctx);
   if (info->indirect) {
      VKCTX(CmdDispatchIndirect)(bs->cmdbuf, zink_resource(info->indirect)->obj->buffer, info->indirect_offset);
      zink_batch_reference_resource_rw(ctx, zink_resource(info->indirect), false);
   } else
      VKCTX(CmdDispatch)(bs->cmdbuf, info->grid[0], info->grid[1], info->grid[2]);
   bs->has_work = true;
   ctx->last_work_was_compute = true;
   /* flush after 30000 draws and dispatches */
   if (!ctx->unordered_blitting && (unlikely(ctx->work_count >= 30000) || ctx->oom_flush))
      pctx->flush(pctx, NULL, 0);
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE, bool BATCH_CHANGED>
static void
init_batch_changed_functions(struct zink_context *ctx, pipe_draw_func draw_vbo_array[2][6][2], pipe_draw_buffers_func draw_vbo_buffers_array[2][6][2], pipe_draw_vertex_state_func draw_state_array[2][6][2][2])
{
   draw_vbo_array[HAS_MULTIDRAW][DYNAMIC_STATE][BATCH_CHANGED] = zink_draw_vbo<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED>;
   draw_vbo_buffers_array[HAS_MULTIDRAW][DYNAMIC_STATE][BATCH_CHANGED] = zink_draw_vbo_buffers<HAS_MULTIDRAW, DYNAMIC_STATE, BATCH_CHANGED>;
   draw_state_array[HAS_MULTIDRAW][DYNAMIC_STATE][0][BATCH_CHANGED] = zink_draw_vertex_state<HAS_MULTIDRAW, DYNAMIC_STATE, POPCNT_NO, BATCH_CHANGED>;
   draw_state_array[HAS_MULTIDRAW][DYNAMIC_STATE][1][BATCH_CHANGED] = zink_draw_vertex_state<HAS_MULTIDRAW, DYNAMIC_STATE, POPCNT_YES, BATCH_CHANGED>;
}

template <zink_multidraw HAS_MULTIDRAW, zink_dynamic_state DYNAMIC_STATE>
static void
init_dynamic_state_functions(struct zink_context *ctx, pipe_draw_func draw_vbo_array[2][6][2], pipe_draw_buffers_func draw_vbo_buffers_array[2][6][2], pipe_draw_vertex_state_func draw_state_array[2][6][2][2])
{
   init_batch_changed_functions<HAS_MULTIDRAW, DYNAMIC_STATE, false>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_batch_changed_functions<HAS_MULTIDRAW, DYNAMIC_STATE, true>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
}

template <zink_multidraw HAS_MULTIDRAW>
static void
init_multidraw_functions(struct zink_context *ctx, pipe_draw_func draw_vbo_array[2][6][2], pipe_draw_buffers_func draw_vbo_buffers_array[2][6][2], pipe_draw_vertex_state_func draw_state_array[2][6][2][2])
{
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_NO_DYNAMIC_STATE>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_DYNAMIC_STATE>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_DYNAMIC_STATE2>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_DYNAMIC_VERTEX_INPUT2>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_DYNAMIC_STATE3>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_dynamic_state_functions<HAS_MULTIDRAW, ZINK_DYNAMIC_VERTEX_INPUT>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
}

static void
init_all_draw_functions(struct zink_context *ctx, pipe_draw_func draw_vbo_array[2][6][2], pipe_draw_buffers_func draw_vbo_buffers_array[2][6][2], pipe_draw_vertex_state_func draw_state_array[2][6][2][2])
{
   init_multidraw_functions<ZINK_NO_MULTIDRAW>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   init_multidraw_functions<ZINK_MULTIDRAW>(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
}

template <bool BATCH_CHANGED>
static void
init_grid_batch_changed_functions(struct zink_context *ctx)
{
   ctx->launch_grid[BATCH_CHANGED] = zink_launch_grid<BATCH_CHANGED>;
}

static void
init_all_grid_functions(struct zink_context *ctx)
{
   init_grid_batch_changed_functions<false>(ctx);
   init_grid_batch_changed_functions<true>(ctx);
}

static void
zink_invalid_draw_vbo(struct pipe_context *pipe,
                      const struct pipe_draw_info *dinfo,
                      unsigned drawid_offset,
                      const struct pipe_draw_indirect_info *dindirect,
                      const struct pipe_draw_start_count_bias *draws,
                      unsigned num_draws)
{
   UNREACHABLE("vertex shader not bound");
}

static void
zink_invalid_draw_mesh_tasks(struct pipe_context *pctx, const struct pipe_grid_info *dinfo)
{
   UNREACHABLE("mesh shader not bound");
}

static void
zink_invalid_draw_vbo_buffers(struct pipe_context *pctx,
                              const struct pipe_draw_info *dinfo,
                              const struct pipe_vertex_buffer *buffers,
                              unsigned buffer_count,
                              const struct pipe_draw_start_count_bias *draws,
                              unsigned num_draws)
{
   UNREACHABLE("vertex shader not bound");
}

static void
zink_invalid_draw_vertex_state(struct pipe_context *pipe,
                               struct pipe_vertex_state *vstate,
                               uint32_t partial_velem_mask,
                               struct pipe_draw_vertex_state_info info,
                               const struct pipe_draw_start_count_bias *draws,
                               unsigned num_draws)
{
   UNREACHABLE("vertex shader not bound");
}

static void
zink_invalid_launch_grid(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
   UNREACHABLE("compute shader not bound");
}

#define STAGE_BASE 0
#define STAGE_BASE_GS (BITFIELD_BIT(MESA_SHADER_GEOMETRY) >> 1)
#define STAGE_BASE_TES (BITFIELD_BIT(MESA_SHADER_TESS_EVAL) >> 1)
#define STAGE_BASE_TES_GS ((BITFIELD_BIT(MESA_SHADER_TESS_EVAL) | BITFIELD_BIT(MESA_SHADER_GEOMETRY)) >> 1)
#define STAGE_BASE_TCS_TES ((BITFIELD_BIT(MESA_SHADER_TESS_CTRL) | BITFIELD_BIT(MESA_SHADER_TESS_EVAL)) >> 1)
#define STAGE_BASE_TCS_TES_GS ((BITFIELD_BIT(MESA_SHADER_TESS_CTRL) | BITFIELD_BIT(MESA_SHADER_TESS_EVAL) | BITFIELD_BIT(MESA_SHADER_GEOMETRY)) >> 1)

template <unsigned STAGE_MASK>
static uint32_t
hash_gfx_program(const void *key)
{
   const struct zink_shader **shaders = (const struct zink_shader**)key;
   uint32_t base_hash = shaders[MESA_SHADER_VERTEX]->hash ^ shaders[MESA_SHADER_FRAGMENT]->hash;
   if (STAGE_MASK == STAGE_BASE) //VS+FS
      return base_hash;
   if (STAGE_MASK == STAGE_BASE_GS) //VS+GS+FS
      return base_hash ^ shaders[MESA_SHADER_GEOMETRY]->hash;
   /*VS+TCS+FS isn't a thing */
   /*VS+TCS+GS+FS isn't a thing */
   if (STAGE_MASK == STAGE_BASE_TES) //VS+TES+FS
      return base_hash ^ shaders[MESA_SHADER_TESS_EVAL]->hash;
   if (STAGE_MASK == STAGE_BASE_TES_GS) //VS+TES+GS+FS
      return base_hash ^ shaders[MESA_SHADER_GEOMETRY]->hash ^ shaders[MESA_SHADER_TESS_EVAL]->hash;
   if (STAGE_MASK == STAGE_BASE_TCS_TES) //VS+TCS+TES+FS
      return base_hash ^ shaders[MESA_SHADER_TESS_CTRL]->hash ^ shaders[MESA_SHADER_TESS_EVAL]->hash;

   /* all stages */
   return base_hash ^ shaders[MESA_SHADER_GEOMETRY]->hash ^ shaders[MESA_SHADER_TESS_CTRL]->hash ^ shaders[MESA_SHADER_TESS_EVAL]->hash;
}

template <unsigned STAGE_MASK>
static bool
equals_gfx_program(const void *a, const void *b)
{
   const void **sa = (const void**)a;
   const void **sb = (const void**)b;
   STATIC_ASSERT(MESA_SHADER_VERTEX == 0);
   STATIC_ASSERT(MESA_SHADER_TESS_CTRL == 1);
   STATIC_ASSERT(MESA_SHADER_TESS_EVAL == 2);
   STATIC_ASSERT(MESA_SHADER_GEOMETRY == 3);
   STATIC_ASSERT(MESA_SHADER_FRAGMENT == 4);
   if (STAGE_MASK == STAGE_BASE) //VS+FS
      return sa[MESA_SHADER_VERTEX] == sb[MESA_SHADER_VERTEX] &&
             sa[MESA_SHADER_FRAGMENT] == sb[MESA_SHADER_FRAGMENT];
   if (STAGE_MASK == STAGE_BASE_GS) //VS+GS+FS
      return sa[MESA_SHADER_VERTEX] == sb[MESA_SHADER_VERTEX] &&
             !memcmp(&sa[MESA_SHADER_GEOMETRY], &sb[MESA_SHADER_GEOMETRY], sizeof(void*) * 2);
   /*VS+TCS+FS isn't a thing */
   /*VS+TCS+GS+FS isn't a thing */
   if (STAGE_MASK == STAGE_BASE_TES) //VS+TES+FS
      return sa[MESA_SHADER_VERTEX] == sb[MESA_SHADER_VERTEX] &&
             sa[MESA_SHADER_TESS_EVAL] == sb[MESA_SHADER_TESS_EVAL] &&
             sa[MESA_SHADER_FRAGMENT] == sb[MESA_SHADER_FRAGMENT];
   if (STAGE_MASK == STAGE_BASE_TES_GS) //VS+TES+GS+FS
      return sa[MESA_SHADER_VERTEX] == sb[MESA_SHADER_VERTEX] &&
             !memcmp(&sa[MESA_SHADER_TESS_EVAL], &sb[MESA_SHADER_TESS_EVAL], sizeof(void*) * 3);
   if (STAGE_MASK == STAGE_BASE_TCS_TES) //VS+TCS+TES+FS
      return !memcmp(sa, sb, sizeof(void*) * 3) &&
             sa[MESA_SHADER_FRAGMENT] == sb[MESA_SHADER_FRAGMENT];

   /* all stages */
   return !memcmp(a, b, sizeof(void*) * ZINK_GFX_SHADER_COUNT);
}

template <unsigned HAS_TASK>
static uint32_t
hash_mesh_program(const void *key)
{
   const struct zink_shader **shaders = (const struct zink_shader**)key;
   uint32_t base_hash = shaders[MESA_SHADER_MESH]->hash ^ shaders[MESA_SHADER_FRAGMENT]->hash;
   if (HAS_TASK == 0) //MS+FS
      return base_hash;
   //TS+MS+FS
   return base_hash ^ shaders[MESA_SHADER_TASK]->hash;
}

template <unsigned HAS_TASK>
static bool
equals_mesh_program(const void *a, const void *b)
{
   const void **sa = (const void**)a;
   const void **sb = (const void**)b;
   STATIC_ASSERT(MESA_SHADER_TASK == 6);
   STATIC_ASSERT(MESA_SHADER_MESH == 7);
   STATIC_ASSERT(MESA_SHADER_FRAGMENT == 4);
   if (HAS_TASK == 0) //MS+FS
      return sa[MESA_SHADER_MESH] == sb[MESA_SHADER_MESH] &&
             sa[MESA_SHADER_FRAGMENT] == sb[MESA_SHADER_FRAGMENT];
   //TS+MS+FS
   return sa[MESA_SHADER_FRAGMENT] == sb[MESA_SHADER_FRAGMENT] &&
          !memcmp(&sa[MESA_SHADER_TASK], &sb[MESA_SHADER_TASK], sizeof(void*) * 2);
}

extern "C"
void
zink_init_draw_functions(struct zink_context *ctx, struct zink_screen *screen)
{
   pipe_draw_func draw_vbo_array[2][6] //multidraw, zink_dynamic_state
                                [2];   //batch changed
   pipe_draw_buffers_func draw_vbo_buffers_array[2][6] //multidraw, zink_dynamic_state
                                                [2];   //batch changed
   pipe_draw_vertex_state_func draw_state_array[2][6] //multidraw, zink_dynamic_state
                                               [2][2];   //has_popcnt, batch changed
   zink_dynamic_state dynamic;
   if (screen->info.have_EXT_extended_dynamic_state) {
      if (screen->info.have_EXT_extended_dynamic_state2) {
         if (screen->info.have_EXT_extended_dynamic_state3) {
            if (screen->info.have_EXT_vertex_input_dynamic_state)
               dynamic = ZINK_DYNAMIC_VERTEX_INPUT;
            else
               dynamic = ZINK_DYNAMIC_STATE3;
         } else {
            if (screen->info.have_EXT_vertex_input_dynamic_state)
               dynamic = ZINK_DYNAMIC_VERTEX_INPUT2;
            else
               dynamic = ZINK_DYNAMIC_STATE2;
         }
      } else {
         dynamic = ZINK_DYNAMIC_STATE;
      }
   } else {
      dynamic = ZINK_NO_DYNAMIC_STATE;
   }
   init_all_draw_functions(ctx, draw_vbo_array, draw_vbo_buffers_array, draw_state_array);
   memcpy(ctx->draw_vbo, &draw_vbo_array[screen->info.have_EXT_multi_draw]
                                        [dynamic],
                                        sizeof(ctx->draw_vbo));
   memcpy(ctx->draw_vbo_buffers, &draw_vbo_buffers_array[screen->info.have_EXT_multi_draw]
                                                [dynamic],
                                                 sizeof(ctx->draw_vbo_buffers));
   memcpy(ctx->draw_state, &draw_state_array[screen->info.have_EXT_multi_draw]
                                          [dynamic][util_get_cpu_caps()->has_popcnt],
                                          sizeof(ctx->draw_state));

   /* Bind a fake draw_vbo, so that draw_vbo isn't NULL, which would skip
    * initialization of callbacks in upper layers (such as u_threaded_context).
    */
   ctx->base.draw_vbo = zink_invalid_draw_vbo;
   ctx->base.draw_vbo_buffers = zink_invalid_draw_vbo_buffers;
   ctx->base.draw_vertex_state = zink_invalid_draw_vertex_state;

   _mesa_hash_table_init(&ctx->program_cache[0], ctx, hash_gfx_program<0>, equals_gfx_program<0>);
   _mesa_hash_table_init(&ctx->program_cache[1], ctx, hash_gfx_program<1>, equals_gfx_program<1>);
   _mesa_hash_table_init(&ctx->program_cache[2], ctx, hash_gfx_program<2>, equals_gfx_program<2>);
   _mesa_hash_table_init(&ctx->program_cache[3], ctx, hash_gfx_program<3>, equals_gfx_program<3>);
   _mesa_hash_table_init(&ctx->program_cache[4], ctx, hash_gfx_program<4>, equals_gfx_program<4>);
   _mesa_hash_table_init(&ctx->program_cache[5], ctx, hash_gfx_program<5>, equals_gfx_program<5>);
   _mesa_hash_table_init(&ctx->program_cache[6], ctx, hash_gfx_program<6>, equals_gfx_program<6>);
   _mesa_hash_table_init(&ctx->program_cache[7], ctx, hash_gfx_program<7>, equals_gfx_program<7>);
   for (unsigned i = 0; i < ARRAY_SIZE(ctx->program_lock); i++)
      simple_mtx_init(&ctx->program_lock[i], mtx_plain);

   ctx->base.draw_mesh_tasks = zink_draw_mesh_tasks<true>;
   _mesa_hash_table_init(&ctx->mesh_cache[0], ctx, hash_mesh_program<0>, equals_mesh_program<0>);
   _mesa_hash_table_init(&ctx->mesh_cache[1], ctx, hash_mesh_program<1>, equals_mesh_program<1>);
   for (unsigned i = 0; i < ARRAY_SIZE(ctx->mesh_lock); i++)
      simple_mtx_init(&ctx->mesh_lock[i], mtx_plain);
   ctx->draw_mesh_tasks[0] = zink_draw_mesh_tasks<false>;
   ctx->draw_mesh_tasks[1] = zink_draw_mesh_tasks<true>;
   ctx->base.draw_mesh_tasks = zink_invalid_draw_mesh_tasks;
}

void
zink_init_grid_functions(struct zink_context *ctx)
{
   init_all_grid_functions(ctx);
   /* Bind a fake launch_grid, so that draw_vbo isn't NULL, which would skip
    * initialization of callbacks in upper layers (such as u_threaded_context).
    */
   ctx->base.launch_grid = zink_invalid_launch_grid;
}

void
zink_init_screen_pipeline_libs(struct zink_screen *screen)
{
   _mesa_set_init(&screen->pipeline_libs[0], NULL, hash_gfx_program<0>, equals_gfx_program<0>);
   _mesa_set_init(&screen->pipeline_libs[1], NULL, hash_gfx_program<1>, equals_gfx_program<1>);
   _mesa_set_init(&screen->pipeline_libs[2], NULL, hash_gfx_program<2>, equals_gfx_program<2>);
   _mesa_set_init(&screen->pipeline_libs[3], NULL, hash_gfx_program<3>, equals_gfx_program<3>);
   _mesa_set_init(&screen->pipeline_libs[4], NULL, hash_gfx_program<4>, equals_gfx_program<4>);
   _mesa_set_init(&screen->pipeline_libs[5], NULL, hash_gfx_program<5>, equals_gfx_program<5>);
   _mesa_set_init(&screen->pipeline_libs[6], NULL, hash_gfx_program<6>, equals_gfx_program<6>);
   _mesa_set_init(&screen->pipeline_libs[7], NULL, hash_gfx_program<7>, equals_gfx_program<7>);
   for (unsigned i = 0; i < ARRAY_SIZE(screen->pipeline_libs_lock); i++)
      simple_mtx_init(&screen->pipeline_libs_lock[i], mtx_plain);

   _mesa_set_init(&screen->mesh_pipeline_libs[0], NULL, hash_mesh_program<0>, equals_mesh_program<0>);
   _mesa_set_init(&screen->mesh_pipeline_libs[1], NULL, hash_mesh_program<1>, equals_mesh_program<1>);
   for (unsigned i = 0; i < ARRAY_SIZE(screen->mesh_pipeline_libs_lock); i++)
      simple_mtx_init(&screen->mesh_pipeline_libs_lock[i], mtx_plain);
}
