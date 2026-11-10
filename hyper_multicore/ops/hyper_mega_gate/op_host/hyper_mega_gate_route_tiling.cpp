/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route_tiling.h"
#include "route_tiling_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

using namespace AscendC;  // NOLINT(build/namespaces) TopKMode lives in AscendC

constexpr uint32_t kRouteStageCount = 10;
constexpr uint32_t kExpertAlignment = 64;
constexpr uint32_t kSelectedExpertAlignment = 8;
bool HasValidRanks(const gert::Shape &logits, const gert::Shape &text_bias, const gert::Shape &vision_bias,
                   const gert::Shape &image_mask, const gert::Shape &runtime, const gert::Shape &profile) {
  return logits.GetDimNum() == 2 && text_bias.GetDimNum() == 1 && vision_bias.GetDimNum() == 1 &&
         image_mask.GetDimNum() == 1 && runtime.GetDimNum() == 1 && profile.GetDimNum() == 1;
}

bool HasValidDimensions(const gert::Shape &logits, const gert::Shape &text_bias, const gert::Shape &vision_bias,
                        const gert::Shape &image_mask, const gert::Shape &runtime, const gert::Shape &profile,
                        int64_t mask_elements, int64_t top_k) {
  return logits.GetDim(0) > 0 && logits.GetDim(0) <= std::numeric_limits<uint32_t>::max() && logits.GetDim(1) > 0 &&
         logits.GetDim(1) <= std::numeric_limits<uint32_t>::max() && text_bias.GetDim(0) == logits.GetDim(1) &&
         vision_bias.GetDim(0) == logits.GetDim(1) && image_mask.GetDim(0) == mask_elements && runtime.GetDim(0) > 0 &&
         profile.GetDim(0) > 0 && top_k > 0 && top_k <= logits.GetDim(1);
}

ge::graphStatus ParseInputs(gert::TilingContext *context, RouteInputs &inputs) {
  const auto *logits_shape = context->GetInputShape(0);
  const auto *text_bias_shape = context->GetInputShape(1);
  const auto *vision_bias_shape = context->GetInputShape(2);
  const auto *image_mask_shape = context->GetInputShape(3);
  const auto *runtime_shape = context->GetInputShape(4);
  const auto *profile_shape = context->GetInputShape(5);
  OP_CHECK_NULL_WITH_CONTEXT(context, logits_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, text_bias_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, vision_bias_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, image_mask_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, runtime_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, profile_shape);
  const auto &logits = logits_shape->GetStorageShape();
  const auto &text_bias = text_bias_shape->GetStorageShape();
  const auto &vision_bias = vision_bias_shape->GetStorageShape();
  const auto &image_mask = image_mask_shape->GetStorageShape();
  const auto &runtime = runtime_shape->GetStorageShape();
  const auto &profile = profile_shape->GetStorageShape();
  auto *attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t *top_k = attrs->GetAttrPointer<int64_t>(0);
  const float *scaling = attrs->GetAttrPointer<float>(1);
  const bool *use_vision_bias = attrs->GetAttrPointer<bool>(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, top_k);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaling);
  OP_CHECK_NULL_WITH_CONTEXT(context, use_vision_bias);
  const int64_t expected_mask_elements = *use_vision_bias ? logits.GetDim(0) : 1;
  if (!HasValidRanks(logits, text_bias, vision_bias, image_mask, runtime, profile) ||
      !HasValidDimensions(logits, text_bias, vision_bias, image_mask, runtime, profile, expected_mask_elements,
                          *top_k) ||
      !std::isfinite(*scaling)) {
    const auto &logits_origin = logits_shape->GetOriginShape();
    const auto &text_bias_origin = text_bias_shape->GetOriginShape();
    OP_LOGE(context->GetNodeName(),
            "Invalid HyperMegaGateRoute shape or attributes: logits storage=[%ld,%ld] rank=%zu, "
            "origin=[%ld,%ld] rank=%zu, text_bias storage=[%ld] rank=%zu, origin=[%ld] rank=%zu, "
            "vision_bias=[%ld], image_mask=[%ld], use_vision_bias=%d, top_k=%ld, scaling=%f.",
            logits.GetDimNum() > 0 ? logits.GetDim(0) : -1, logits.GetDimNum() > 1 ? logits.GetDim(1) : -1,
            logits.GetDimNum(), logits_origin.GetDimNum() > 0 ? logits_origin.GetDim(0) : -1,
            logits_origin.GetDimNum() > 1 ? logits_origin.GetDim(1) : -1, logits_origin.GetDimNum(),
            text_bias.GetDimNum() > 0 ? text_bias.GetDim(0) : -1, text_bias.GetDimNum(),
            text_bias_origin.GetDimNum() > 0 ? text_bias_origin.GetDim(0) : -1, text_bias_origin.GetDimNum(),
            vision_bias.GetDimNum() > 0 ? vision_bias.GetDim(0) : -1,
            image_mask.GetDimNum() > 0 ? image_mask.GetDim(0) : -1, static_cast<int>(*use_vision_bias), *top_k,
            *scaling);
    return ge::GRAPH_FAILED;
  }
  inputs.tokens = static_cast<uint32_t>(logits.GetDim(0));
  inputs.expert_count = static_cast<uint32_t>(logits.GetDim(1));
  inputs.top_k = static_cast<uint32_t>(*top_k);
  const auto *use_sigmoid = attrs->GetAttrPointer<bool>(3);
  const auto *normalize = attrs->GetAttrPointer<bool>(4);
  const auto *use_bias = attrs->GetAttrPointer<bool>(5);
  OP_CHECK_NULL_WITH_CONTEXT(context, use_sigmoid);
  OP_CHECK_NULL_WITH_CONTEXT(context, normalize);
  OP_CHECK_NULL_WITH_CONTEXT(context, use_bias);
  inputs.use_sigmoid = *use_sigmoid;
  inputs.normalize = *normalize;
  inputs.use_bias = *use_bias;
  inputs.scaling = *scaling;
  inputs.use_vision_bias = *use_vision_bias;
  inputs.runtime_bytes = static_cast<uint64_t>(runtime.GetDim(0));
  inputs.profile_bytes = static_cast<uint64_t>(profile.GetDim(0));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFunc(gert::TilingContext *context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  RouteInputs inputs{};
  if (ParseInputs(context, inputs) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const uint32_t vector_cores = platform.GetCoreNumAiv();
  if (vector_cores == 0) {
    OP_LOGE(context->GetNodeName(), "HyperMegaGateRoute requires AIV cores.");
    return ge::GRAPH_FAILED;
  }
  const uint32_t worker_capacity = std::min(inputs.tokens, vector_cores);
  const uint32_t rows_per_worker = CeilDiv(inputs.tokens, worker_capacity);
  const uint32_t active_workers = CeilDiv(inputs.tokens, rows_per_worker);
  const uint32_t e_align = CeilDiv(inputs.expert_count, kExpertAlignment) * kExpertAlignment;
  const uint32_t k_align = CeilDiv(inputs.top_k, kSelectedExpertAlignment) * kSelectedExpertAlignment;
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  uint32_t batch = 0;
  uint32_t shared_tmp = 0;
  if (!SelectBatchRows(context, platform, inputs, rows_per_worker, e_align, k_align, vector_cores, ub_size, batch,
                       shared_tmp)) {
    return ge::GRAPH_FAILED;
  }

  HyperMegaGateRouteTilingData tiling;
  tiling.set_tokenCount(inputs.tokens);
  tiling.set_expertCount(inputs.expert_count);
  tiling.set_topK(inputs.top_k);
  tiling.set_routedScalingFactor(inputs.scaling);
  tiling.set_activeAivWorkerCount(active_workers);
  tiling.set_rowsPerWorker(rows_per_worker);
  tiling.set_aivWorkerSlotCapacity(vector_cores);
  tiling.set_batchRows(batch);
  tiling.set_expertAlign(e_align);
  tiling.set_kAlign(k_align);
  tiling.set_sharedTmpBytes(shared_tmp);
  tiling.set_stageCount(kRouteStageCount);
  tiling.set_useVisionBias(inputs.use_vision_bias ? 1U : 0U);
  tiling.set_useSigmoid(inputs.use_sigmoid ? 1U : 0U);
  tiling.set_normalize(inputs.normalize ? 1U : 0U);
  tiling.set_useBias(inputs.use_bias ? 1U : 0U);
  tiling.set_runtimeConfigBytes(inputs.runtime_bytes);
  tiling.set_diagnosticBufferBytes(inputs.profile_bytes);
  uint64_t route_workspace = 0;
  const uint64_t score_bytes = static_cast<uint64_t>(inputs.tokens) * inputs.expert_count * sizeof(float);
  const uint64_t selected_float_bytes = static_cast<uint64_t>(inputs.tokens) * inputs.top_k * sizeof(float);
  const uint64_t selected_i32_bytes = static_cast<uint64_t>(inputs.tokens) * inputs.top_k * sizeof(int32_t);
  const uint64_t row_bytes = static_cast<uint64_t>(inputs.tokens) * sizeof(float);
  tiling.set_scoreAOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + score_bytes, kWorkspaceAlignment);
  tiling.set_topkValuesOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + selected_float_bytes, kWorkspaceAlignment);
  tiling.set_indicesI32Offset(route_workspace);
  route_workspace = AlignUp(route_workspace + selected_i32_bytes, kWorkspaceAlignment);
  tiling.set_rowSumOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + row_bytes, kWorkspaceAlignment);
  tiling.set_routeWorkspaceBytes(route_workspace);
  if (!TopKTilingFunc(platform, e_align, batch, inputs.top_k, sizeof(float), true, TopKMode::TOPK_NORMAL, true,
                      tiling.topkTiling)) {
    OP_LOGE(context->GetNodeName(),
            "TopKTilingFunc failed: tokens=%u experts=%ld topK=%ld batchRows=%u availableAivWorkers=%u.", inputs.tokens,
            static_cast<int64_t>(inputs.expert_count), static_cast<int64_t>(inputs.top_k), batch, vector_cores);
    return ge::GRAPH_FAILED;
  }
  OP_LOGD(context->GetNodeName(),
          "HyperMegaGate Route tiling: tokens=%u experts=%ld availableAivWorkers=%u activeAivWorkers=%u "
          "rowsPerWorker=%u batchRows=%u useVisionBias=%d",
          inputs.tokens, static_cast<int64_t>(inputs.expert_count), vector_cores, active_workers, rows_per_worker,
          batch, static_cast<int>(inputs.use_vision_bias));
  auto *raw_tiling = context->GetRawTilingData();
  OP_CHECK_NULL_WITH_CONTEXT(context, raw_tiling);
  tiling.SaveToBuffer(raw_tiling->GetData(), raw_tiling->GetCapacity());
  raw_tiling->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(active_workers);
  context->SetTilingKey(0);
  size_t *workspace_sizes = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace_sizes);
  workspace_sizes[0] = platform.GetLibApiWorkSpaceSize() + route_workspace;
  return ge::GRAPH_SUCCESS;
}

struct HyperMegaGateRouteCompileInfo {};

ge::graphStatus TilingPrepare(gert::TilingParseContext *context) {
  return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(HyperMegaGateRoute).Tiling(TilingFunc).TilingParse<HyperMegaGateRouteCompileInfo>(TilingPrepare);

}  // namespace optiling
