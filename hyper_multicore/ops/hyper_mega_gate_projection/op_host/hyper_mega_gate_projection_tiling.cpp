/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_projection_tiling.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "../../hyper_mega_gate/op_host/route_tiling_utils.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
using namespace AscendC;
bool HasValidProjectionShape(const gert::Shape& x, const gert::Shape& w) {
  if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || x.GetDim(1) != w.GetDim(1) || x.GetDim(0) <= 0 ||
      x.GetDim(0) > std::numeric_limits<int32_t>::max() || x.GetDim(1) <= 0 ||
      x.GetDim(1) > std::numeric_limits<int32_t>::max() || x.GetDim(1) % 64 || (w.GetDim(0) < 8 || w.GetDim(0) > 384))
    return false;
  return true;
}

bool SetProjectionAttributes(gert::TilingContext* ctx, HyperMegaGateProjectionTilingData& data) {
  constexpr int64_t expected[] = {128, 128, 64, 128, 128, 256, 32, 1};
  for (size_t index = 0; index < 8; ++index) {
    const auto* value = ctx->GetAttrs()->GetAttrPointer<int64_t>(index);
    if (value == nullptr || *value != expected[index]) return false;
  }
  const auto* profile = ctx->GetAttrs()->GetAttrPointer<int64_t>(8);
  const auto* use_bias = ctx->GetAttrs()->GetAttrPointer<int64_t>(9);
  if (!profile || !use_bias || (*profile != 0 && *profile != 2) || (*use_bias != 0 && *use_bias != 1)) {
    return false;
  }
  data.set_baseM(128);
  data.set_baseN(128);
  data.set_baseK(64);
  data.set_mL1(128);
  data.set_nL1(128);
  data.set_kL1(256);
  data.set_tileRows(32);
  data.set_overlap(1);
  data.set_profile(*profile);
  data.set_useBias(*use_bias);

  return true;
}

static ge::graphStatus Tiling(gert::TilingContext* ctx) {
  if (!ctx || !ctx->GetInputShape(0) || !ctx->GetInputShape(1) || !ctx->GetAttrs() || !ctx->GetRawTilingData() ||
      !ctx->GetPlatformInfo())
    return ge::GRAPH_FAILED;
  const auto& x = ctx->GetInputShape(0)->GetStorageShape();
  const auto& w = ctx->GetInputShape(1)->GetStorageShape();
  if (!HasValidProjectionShape(x, w)) return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
  // The ready flags and diagnostic ABI use this explicit hardware contract.
  if (platform.GetCoreNumAic() != 32 || platform.GetCoreNumAiv() != 64) return ge::GRAPH_FAILED;
  HyperMegaGateProjectionTilingData data;
  data.set_rows(x.GetDim(0));
  data.set_columns(w.GetDim(0));
  data.set_depth(x.GetDim(1));
  data.set_workers(platform.GetCoreNumAic());
  data.set_vectorWorkers(platform.GetCoreNumAiv());
  if (!SetProjectionAttributes(ctx, data)) return ge::GRAPH_FAILED;
  RouteInputs inputs{};
  inputs.tokens = x.GetDim(0);
  inputs.expert_count = w.GetDim(0);
  inputs.top_k = 8;
  inputs.use_sigmoid = true;
  inputs.use_bias = data.get_useBias();
  inputs.normalize = true;
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  uint32_t batch = 0, shared = 0;
  const uint32_t expert_align = CeilDiv(inputs.expert_count, 64) * 64;
  if (!SelectBatchRows(ctx, platform, inputs, CeilDiv(data.get_tileRows(), 2), expert_align, 8,
                       data.get_vectorWorkers(), ub, batch, shared))
    return ge::GRAPH_FAILED;
  data.set_batchRows(batch);
  data.set_expertAlign(expert_align);
  data.set_kAlign(8);
  data.set_sharedTmpBytes(shared);
  uint64_t bytes = 0;
  data.set_scoreAOffset(bytes);
  bytes = AlignUp(bytes + uint64_t(data.get_rows()) * inputs.expert_count * 4, 512);
  data.set_topkValuesOffset(bytes);
  bytes = AlignUp(bytes + uint64_t(data.get_rows()) * 8 * 4, 512);
  data.set_indicesI32Offset(bytes);
  bytes = AlignUp(bytes + uint64_t(data.get_rows()) * 8 * 4, 512);
  data.set_rowSumOffset(bytes);
  bytes = AlignUp(bytes + uint64_t(data.get_rows()) * 4, 512);
  if (!TopKTilingFunc(platform, expert_align, batch, 8, 4, true, TopKMode::TOPK_NORMAL, true, data.topkTiling))
    return ge::GRAPH_FAILED;
  ctx->SetBlockDim(data.get_workers());
  data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
  ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
  ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + bytes;
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperMegaGateProjection).Tiling(Tiling);
}  // namespace optiling
