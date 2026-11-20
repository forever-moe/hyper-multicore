/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H
#define HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H

#include "runtime/aiv_pipeline_worker.h"

template <typename TilingData>
__aicore__ inline bool IsRouteGradRuntimeValid(__gm__ uint8_t *runtime, const TilingData &tiling) {
  if (!MulticoreRuntime::isAivPipelineStorageValid(runtime, tiling.runtimeConfigBytes,
                                                  tiling.diagnosticBufferBytes)) return false;
  return tiling.activeAivWorkerCount > 0 && tiling.activeAivWorkerCount <= tiling.aivWorkerSlotCapacity &&
         tiling.activeAivWorkerCount <= MulticoreRuntime::getNumWorkers(runtime) && tiling.stageCount > 0 &&
         MulticoreRuntime::getTaskNum(runtime) == tiling.stageCount &&
         MulticoreRuntime::getTaskIndexNumByTaskType(runtime, MulticoreRuntime::TaskAiCoreType::TASK_AICORE_VECTOR) ==
           static_cast<int32_t>(tiling.stageCount);
}

#endif  // HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H
