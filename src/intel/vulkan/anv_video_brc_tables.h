/*
 * Copyright (c) 2026 Igalia S.L.
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

#ifndef ANV_VIDEO_BRC_TABLES_H
#define ANV_VIDEO_BRC_TABLES_H

#include <stdint.h>

/* H.264 BRC tables */
extern const int8_t anv_h264_brc_global_rate_qp_adj_i[64];
extern const int8_t anv_h264_brc_global_rate_qp_adj_p[64];
extern const int8_t anv_h264_brc_global_rate_qp_adj_b[64];
extern const uint8_t anv_h264_brc_dist_threshld_i[10];
extern const uint8_t anv_h264_brc_dist_threshld_p[10];
extern const int8_t anv_h264_brc_cbr_dist_qp_adj_i[81];
extern const int8_t anv_h264_brc_cbr_dist_qp_adj_p[81];
extern const int8_t anv_h264_brc_cbr_dist_qp_adj_b[81];
extern const int8_t anv_h264_brc_vbr_dist_qp_adj_i[81];
extern const int8_t anv_h264_brc_vbr_dist_qp_adj_p[81];
extern const int8_t anv_h264_brc_vbr_dist_qp_adj_b[81];
extern const int8_t anv_h264_brc_cbr_frm_sz_adj_i[72];
extern const int8_t anv_h264_brc_cbr_frm_sz_adj_p[72];
extern const int8_t anv_h264_brc_cbr_frm_sz_adj_b[72];
extern const int8_t anv_h264_brc_vbr_frm_sz_adj_i[72];
extern const int8_t anv_h264_brc_vbr_frm_sz_adj_p[72];
extern const int8_t anv_h264_brc_vbr_frm_sz_adj_b[72];
extern const uint8_t anv_h264_brc_frm_sz_min_i[9];
extern const uint8_t anv_h264_brc_frm_sz_min_p[9];
extern const uint8_t anv_h264_brc_frm_sz_max_i[9];
extern const uint8_t anv_h264_brc_frm_sz_max_p[9];
extern const uint8_t anv_h264_brc_frm_sz_scg_i[9];
extern const uint8_t anv_h264_brc_frm_sz_scg_p[9];
extern const uint8_t anv_h264_brc_cost_i_intra_non_pred[42];
extern const uint8_t anv_h264_brc_cost_i_intra8x8[42];
extern const uint8_t anv_h264_brc_cost_i_intra4x4[42];
extern const uint8_t anv_h264_brc_cost_p_intra_non_pred[42];
extern const uint8_t anv_h264_brc_cost_p_intra16x16[42];
extern const uint8_t anv_h264_brc_cost_p_intra8x8[42];
extern const uint8_t anv_h264_brc_cost_p_intra4x4[42];
extern const uint8_t anv_h264_brc_cost_p_inter16x8[42];
extern const uint8_t anv_h264_brc_cost_p_inter8x8[42];
extern const uint8_t anv_h264_brc_cost_p_inter16x16[42];
extern const uint8_t anv_h264_brc_cost_p_ref_id[42];
extern const uint32_t anv_h264_brc_hme_cost[8][52];

/* H.264 BRC tables (above gen12.5) */
extern const uint8_t h264_vdenc_brc_huc_const_reserved[5][630];

/* H.265 BRC tables */
extern const uint8_t anv_hevc_brc_est_rate_thresh[7];
extern const int8_t anv_hevc_brc_inst_rate_thresh_p[4];
extern const int8_t anv_hevc_brc_inst_rate_thresh_b[4];
extern const int8_t anv_hevc_brc_inst_rate_thresh_i[4];
extern const uint16_t anv_hevc_brc_start_gadj_frame[4];
extern const uint8_t anv_hevc_brc_start_gadj_mult[5];
extern const uint8_t anv_hevc_brc_start_gadj_div[5];
extern const uint8_t anv_hevc_brc_rate_ratio_threshold[7];
extern const uint8_t anv_hevc_brc_rate_ratio_threshold_qp[8];
extern const double anv_hevc_brc_dev_thresh_i_neg[4];
extern const double anv_hevc_brc_dev_thresh_i_pos[4];
extern const double anv_hevc_brc_dev_thresh_pb_neg[4];
extern const double anv_hevc_brc_dev_thresh_pb_pos[4];
extern const double anv_hevc_brc_dev_thresh_vbr_neg[4];
extern const double anv_hevc_brc_dev_thresh_vbr_pos[4];
extern const int8_t anv_hevc_brc_lowdelay_dev_thresh_pb[8];
extern const int8_t anv_hevc_brc_lowdelay_dev_thresh_vbr[8];
extern const int8_t anv_hevc_brc_lowdelay_dev_thresh_i[8];
extern const uint16_t anv_hevc_brc_sad_qp_lambda_i[52];
extern const uint16_t anv_hevc_brc_sad_qp_lambda_p[52];
extern const uint16_t anv_hevc_brc_rd_qp_lambda_i[52];
extern const uint16_t anv_hevc_brc_rd_qp_lambda_p[52];
extern const uint8_t anv_hevc_brc_penalty_intra_non_dc32[52];
extern const uint32_t anv_hevc_brc_mode_costs_i[364];
extern const uint32_t anv_hevc_brc_mode_costs_pb[364];
extern const uint32_t anv_hevc_brc_const_tables[900];

#endif /* ANV_VIDEO_BRC_TABLES_H */
