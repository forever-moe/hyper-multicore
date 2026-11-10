/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <algorithm>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace {
using namespace AscendC;
constexpr uint32_t kMaximumBatchRows = 256;
constexpr uint64_t kWorkspaceAlignment = 512;

uint32_t CeilDiv(uint32_t value, uint32_t factor) { return value / factor + (value % factor != 0); }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

struct RouteInputs {
  uint32_t tokens;
  uint32_t expert_count;
  uint32_t top_k;
  float scaling;
  bool use_vision_bias;
  bool use_sigmoid;
  bool normalize;
  bool use_bias;
  uint64_t runtime_bytes;
  uint64_t profile_bytes;
};

bool SelectBatchRows(gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform,
                     const RouteInputs& inputs, uint32_t rows_per_worker, uint32_t e_align, uint32_t k_align,
                     uint32_t available_workers, uint64_t ub_size, uint32_t& batch, uint32_t& shared_tmp) {
  batch = std::min(rows_per_worker, kMaximumBatchRows);
  const bool batched_topk = inputs.use_sigmoid && inputs.expert_count == 384 && inputs.top_k == 8 &&
                            platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  uint32_t topk_tmp = 0;
  while (batch > 0) {
    uint32_t sum_tmp_max = 0;
    if (inputs.top_k > 1) {
      uint32_t sum_tmp_min = 0;
      if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        GetReduceSumMaxMinTmpSize(ge::Shape({batch, k_align}), ge::DT_FLOAT, ReducePattern::AR, true, false,
                                  sum_tmp_max, sum_tmp_min);
      } else {
        GetSumMaxMinTmpSize(inputs.top_k, sizeof(float), false, sum_tmp_max, sum_tmp_min);
      }
    }
    uint32_t candidate_topk_tmp = 0;
    uint32_t candidate_topk_tmp_min = 0;
    if (!GetTopKMaxMinTmpSize(platform, e_align, batch, false, true, TopKMode::TOPK_NORMAL, true, sizeof(float),
                              candidate_topk_tmp, candidate_topk_tmp_min)) {
      if (batch == 1) {
        OP_LOGE(context->GetNodeName(),
                "GetTopKMaxMinTmpSize failed: tokens=%u experts=%ld topK=%ld batchRows=%u "
                "availableAivWorkers=%u.",
                inputs.tokens, static_cast<int64_t>(inputs.expert_count), static_cast<int64_t>(inputs.top_k), batch,
                available_workers);
        return false;
      }
      batch = std::max(1u, batch / 2);
      continue;
    }
    topk_tmp = candidate_topk_tmp;
    uint32_t bcast_tmp_max = 0;
    uint32_t bias_bcast_tmp = 0;
    uint32_t row_bcast_tmp = 0;
    uint32_t mask_bcast_tmp = 0;
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({1, static_cast<int64_t>(e_align)}),
                                    ge::Shape({batch, static_cast<int64_t>(e_align)}), sizeof(float), true,
                                    bcast_tmp_max, bias_bcast_tmp);
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({batch, 1}), ge::Shape({batch, k_align}), sizeof(float), false,
                                    bcast_tmp_max, row_bcast_tmp);
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({1, k_align}), ge::Shape({batch, k_align}), sizeof(float),
                                    false, bcast_tmp_max, mask_bcast_tmp);
    shared_tmp = std::max({sum_tmp_max, bias_bcast_tmp, row_bcast_tmp, mask_bcast_tmp});

    // Mirror every InitBuffer in RouteBatchKernel, including double-buffered
    // queues and both FP32 Softplus predicate masks. The library temporary is
    // the larger of the broadcast/sum scratch and TopK scratch.
    const uint64_t score_queue = 2ULL * batch * e_align * sizeof(float);
    const uint64_t weight_queue = 2ULL * batch * k_align * sizeof(float);
    const uint64_t score_buffers = 2ULL * batch * e_align * sizeof(float);
    const uint64_t k_buffers = static_cast<uint64_t>(batch) * k_align * (8 * sizeof(float) + sizeof(int64_t));
    const uint64_t fixed = e_align * sizeof(int32_t) + CeilDiv(batch, 8) * 8 * sizeof(float) + k_align * sizeof(float);
    const uint64_t predicate = 2ULL * CeilDiv(batch * e_align, 256) * 32ULL;
    const uint64_t vision_buffers =
        inputs.use_vision_bias ? 2ULL * e_align * sizeof(float) + AlignUp(batch * sizeof(bool), 32) : 0;
    const uint64_t total = score_queue + weight_queue + score_buffers + k_buffers + fixed + predicate + shared_tmp +
                           topk_tmp + vision_buffers;
    // TopK owns its padded input, row indices, pair lists and selected outputs within one task tile.
    const uint64_t batched_topk_bytes = 4ULL * batch * e_align * sizeof(float) + 2ULL * batch * 96 * sizeof(float) +
                                        2ULL * batch * k_align * sizeof(float);
    const bool batched_topk_fits = !batched_topk || (batched_topk_bytes <= ub_size / 2 && batch * 12 <= 255);
    if (total <= ub_size / 2 && batched_topk_fits) {
      return true;
    }
    if (batch == 1) {
      OP_LOGE(context->GetNodeName(), "Route buffers exceed available UB.");
      return false;
    }
    batch = std::max(1u, batch / 2);
  }
  return false;
}

}  // namespace
}  // namespace optiling
