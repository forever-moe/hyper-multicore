/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "kernel_operator.h"
#ifdef __DAV_CUBE__
#include "owner_matmul.h"
#include "runtime/cube_tile_pipeline_worker.h"
#else
#include "route_worker.h"
#endif
#include "runtime/cv_tile_handoff.h"
#include "runtime/cycle_trace_recorder.h"
using namespace AscendC;

#ifdef __DAV_CUBE__
class ProjectionWorker : public MulticoreRuntime::CubeTilePipelineWorkerBase<ProjectionWorker> {
 public:
  __aicore__ inline void Init(GM_ADDR x_, GM_ADDR w_, GM_ADDR out_, GM_ADDR workspace_, GM_ADDR trace_,
                              GM_ADDR runtime_, GM_ADDR ready_, const HyperMegaGateProjectionTilingData& data_,
                              MulticoreRuntime::CycleTraceRecorder* recorder_ = nullptr) {
    recorder = recorder_;
    x = x_;
    weight = w_;
    logits = out_;
    workspace = workspace_;
    data = data_;
    worker = GetBlockIdx();
    tiles = (data.rows + data.tileRows - 1) / data.tileRows;
    MulticoreRuntime::CubeTilePipelineWorkerBase<ProjectionWorker>::Init(reinterpret_cast<__gm__ uint8_t*>(runtime_),
                                                                         ready_, tiles, data.workers);
  }
  __aicore__ inline void ExecuteComputeKernel(MulticoreRuntime::TaskType type, uint32_t offset, uint32_t tile) {
    if (type != MulticoreRuntime::TaskType::TASK_MATMUL || offset) Trap();

    const uint32_t first = tile * data.tileRows;
    const uint32_t rows = (data.tileRows < data.rows - first ? data.tileRows : data.rows - first);
    const uint64_t profile_start = recorder ? recorder->Now() : 0;
    MatMulV3BasicTilingData mm;
    mm.m = rows;
    mm.n = data.columns;
    mm.k = data.depth;
    mm.usedCoreNum = 1;
    mm.baseM = data.baseM;
    mm.baseN = data.baseN;
    mm.baseK = data.baseK;
    mm.mL1 = data.mL1;
    mm.nL1 = data.nL1;
    mm.kL1 = data.kL1;
    mm.mTailCnt = 1;
    mm.nTailCnt = 1;
    mm.sliceM = 1;
    mm.srcNdStride = 1;
    mm.l1BufferNum = 2;
    mm.l0cDB = 2;
    mm.ubDB = 1;
    MatmulV3Advanced::OwnerMatMulActKernel<float, float, float, float, Cmct::Gemm::layout::RowMajor,
                                           Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor>(
        x + uint64_t(first) * data.depth * 4, weight, nullptr, logits + uint64_t(first) * data.columns * 4, workspace,
        mm);
    PipeBarrier<PIPE_ALL>();
    if (recorder) {
      recorder->Record(0x30000, 0, tile, tile, profile_start, recorder->Now());
    }
  }

 private:
  GM_ADDR x;
  GM_ADDR weight;
  GM_ADDR logits;
  GM_ADDR workspace;
  HyperMegaGateProjectionTilingData data;
  uint32_t worker = 0, tiles = 0;
  MulticoreRuntime::CycleTraceRecorder* recorder = nullptr;
};
#endif

extern "C" __global__ __aicore__ void hyper_mega_gate_projection(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                 GM_ADDR runtime, GM_ADDR ready, GM_ADDR logits,
                                                                 GM_ADDR weights, GM_ADDR indices, GM_ADDR scores,
                                                                 GM_ADDR selected, GM_ADDR denominator, GM_ADDR trace,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(data, tiling);
  InitSocState();
  const uint32_t tiles = (data.rows + data.tileRows - 1) / data.tileRows;
  MulticoreRuntime::CvTileHandoff handoff;
  handoff.Init(ready, tiles, 1);
  const uint32_t worker = GetBlockIdx();
  MulticoreRuntime::CycleTraceRecorder recorder;
  if (data.profile == 2) {
    auto config = reinterpret_cast<__gm__ uint8_t*>(runtime);
    recorder.Init(worker, trace, MulticoreRuntime::getAicProfileRecordCapacity(config),
                  MulticoreRuntime::getAivProfileRecordCapacity(config),
                  MulticoreRuntime::getCubeWorkerCapacity(config));
  }
#ifdef __DAV_CUBE__
  ProjectionWorker projection;
  projection.Init(x, weight, logits, workspace, trace, runtime, ready, data, data.profile == 2 ? &recorder : nullptr);
  projection.Process();
#else
  // AIC i owns tiles i, i+cubeCount, ...; AIV 2i and 2i+1 consume the two row halves.
  for (uint32_t tile = worker / 2; tile < tiles; tile += data.workers) {
    const uint64_t wait_start = data.profile == 2 ? recorder.Now() : 0;
    if (!handoff.Wait(tile)) {
      Trap();
    }
    if (data.profile == 2) {
      recorder.Record(0x10000, 0, tile, tile, wait_start, recorder.Now());
    }
    const uint32_t tileFirst = tile * data.tileRows;
    const uint32_t tileCount = (data.tileRows < data.rows - tileFirst ? data.tileRows : data.rows - tileFirst);
    const uint32_t shard = (tileCount + 1) / 2;
    const uint32_t offset = (worker % 2) * shard;
    const uint32_t rows = offset < tileCount ? (shard < tileCount - offset ? shard : tileCount - offset) : 0;
    if (rows) {
      RoutePipelineWorker route;
      route.Init(worker, reinterpret_cast<__gm__ uint8_t*>(runtime), runtime, reinterpret_cast<__gm__ float*>(logits),
                 reinterpret_cast<__gm__ float*>(bias), reinterpret_cast<__gm__ float*>(bias), nullptr,
                 reinterpret_cast<__gm__ float*>(weights), reinterpret_cast<__gm__ int64_t*>(indices),
                 reinterpret_cast<__gm__ float*>(scores), reinterpret_cast<__gm__ float*>(selected),
                 reinterpret_cast<__gm__ float*>(denominator), GetUserWorkspace(workspace), data, tileFirst + offset,
                 rows);
      if (data.profile == 2) {
        route.ProcessProfiledTile(recorder, tile);
      } else {
        route.Process();
      }
    }
  }
#endif
}
