/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HyperMegaGateProjectionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, columns);
TILING_DATA_FIELD_DEF(uint32_t, depth);
TILING_DATA_FIELD_DEF(uint32_t, workers);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, mL1);
TILING_DATA_FIELD_DEF(uint32_t, nL1);
TILING_DATA_FIELD_DEF(uint32_t, kL1);
TILING_DATA_FIELD_DEF(uint32_t, tileRows);
TILING_DATA_FIELD_DEF(uint32_t, overlap);
TILING_DATA_FIELD_DEF(uint32_t, profile);
TILING_DATA_FIELD_DEF(uint32_t, useBias);
TILING_DATA_FIELD_DEF(uint32_t, batchRows);
TILING_DATA_FIELD_DEF(uint32_t, expertAlign);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, sharedTmpBytes);
TILING_DATA_FIELD_DEF(uint32_t, vectorWorkers);
TILING_DATA_FIELD_DEF(uint64_t, scoreAOffset);
TILING_DATA_FIELD_DEF(uint64_t, topkValuesOffset);
TILING_DATA_FIELD_DEF(uint64_t, indicesI32Offset);
TILING_DATA_FIELD_DEF(uint64_t, rowSumOffset);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HyperMegaGateProjection, HyperMegaGateProjectionTilingData)
}  // namespace optiling
