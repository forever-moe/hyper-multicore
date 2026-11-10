/** Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "kernel_operator.h"
namespace MulticoreRuntime {
// One cache line per token tile. Every Cube publishes exactly once after full K/E completion.
class CvTileHandoff {
 public:
  __aicore__ inline void Init(GM_ADDR ready, uint32_t tiles, uint32_t cubes) {
    state_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ready), tiles * 32);
    cubes_ = cubes;
  }
  __aicore__ inline void Publish(uint32_t tile) {
    AscendC::PipeBarrier<PIPE_ALL>();
    (void)AscendC::AtomicAdd<int32_t>(state_.GetPhyAddr(tile * 32), 1);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline bool Wait(uint32_t tile) {
    const uint64_t start = AscendC::GetSystemCycle();
    const uint32_t offset = tile * 32;
    while (true) {
      AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(state_[offset]);
      if (state_.GetValue(offset) >= static_cast<int32_t>(cubes_)) break;
      if (AscendC::GetSystemCycle() - start > 100000000) return false;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
  }
  __aicore__ inline int32_t Count(uint32_t tile) { return state_.GetValue(tile * 32); }

 private:
  AscendC::GlobalTensor<int32_t> state_;
  uint32_t cubes_ = 0;
};
}  // namespace MulticoreRuntime
