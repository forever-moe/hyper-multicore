/** Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "cv_tile_handoff.h"
#include "kernel_operator.h"
#include "runtime_config.hpp"
namespace MulticoreRuntime {
// Distribute disjoint complete token tiles round-robin across physical Cubes.
// Each tile owns disjoint GM output; publish only after all local tasks drain.
template <class Derived>
class CubeTilePipelineWorkerBase {
 public:
  __aicore__ inline void Init(__gm__ uint8_t* runtime, GM_ADDR ready, uint32_t tiles, uint32_t cubes) {
    runtime_ = runtime;
    tiles_ = tiles;
    cubes_ = cubes;
    worker_ = AscendC::GetBlockIdx();
    tasks_ = reinterpret_cast<__gm__ int32_t*>(runtime + getCubeTaskIndexsOffset(runtime));
    stages_ = getTaskIndexNumByTaskType(runtime, TaskAiCoreType::TASK_AICORE_CUBE);
    if (stages_ != 1) AscendC::Trap();
    handoff_.Init(ready, tiles, 1);
  }
  __aicore__ inline void Process() {
    for (uint32_t tile = worker_; tile < tiles_; tile += cubes_) {
      for (uint32_t stage = 0; stage < stages_; ++stage) {
        const auto task = tasks_[stage];
        if (task < 0 || static_cast<uint32_t>(task) >= getRuntimeTaskCapacity(runtime_)) AscendC::Trap();
        static_cast<Derived*>(this)->ExecuteComputeKernel(getTaskType(runtime_, task),
                                                          getTaskTilingDataOffset(runtime_, task), tile);
      }
      handoff_.Publish(tile);
    }
  }

 private:
  __gm__ uint8_t* runtime_ = nullptr;
  __gm__ int32_t* tasks_ = nullptr;
  uint32_t tiles_ = 0, stages_ = 0, cubes_ = 0, worker_ = 0;
  CvTileHandoff handoff_;
};
}  // namespace MulticoreRuntime
