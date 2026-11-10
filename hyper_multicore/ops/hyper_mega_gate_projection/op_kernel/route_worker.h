/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include "hyper_mega_gate/op_kernel/route_pipeline.h"
#include "runtime/aiv_pipeline_worker.h"
namespace {

constexpr uint32_t kRouteStageCount = 10;

class RoutePipelineWorker : public MulticoreRuntime::AivPipelineWorkerBase<RoutePipelineWorker> {
 public:
  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t* runtime_config, GM_ADDR diagnostic_buffer,
                              __gm__ float* logits, __gm__ float* text_bias, __gm__ float* vision_bias,
                              __gm__ bool* image_mask, __gm__ float* routing_weights, __gm__ int64_t* expert_indices,
                              __gm__ float* route_scores, __gm__ float* selected_scores,
                              __gm__ float* normalization_denominator, GM_ADDR user_workspace,
                              const HyperMegaGateProjectionTilingData& tiling_data, uint64_t first_row,
                              uint32_t row_count) {
    MulticoreRuntime::AivPipelineWorkerBase<RoutePipelineWorker>::Init(worker_id, runtime_config, diagnostic_buffer);
    route_.Init(logits, text_bias, vision_bias, image_mask, false, routing_weights, expert_indices, route_scores,
                selected_scores, normalization_denominator, user_workspace, tiling_data.columns, 8, 1.0f,
                tiling_data.batchRows, tiling_data.expertAlign, tiling_data.kAlign, tiling_data.sharedTmpBytes,
                tiling_data.scoreAOffset, tiling_data.topkValuesOffset, tiling_data.indicesI32Offset,
                tiling_data.rowSumOffset, &tiling_data.topkTiling, first_row, row_count, true, true,
                tiling_data.useBias != 0);
  }

  __aicore__ inline void ExecuteComputeKernel(MulticoreRuntime::TaskType task_type, uint32_t tiling_offset) {
    const uint32_t stage =
        static_cast<uint32_t>(task_type) - static_cast<uint32_t>(MulticoreRuntime::TaskType::TASK_GATE_SOFTPLUS);
    if (stage != tiling_offset || stage >= kRouteStageCount) {
      Trap();
    }
    switch (task_type) {
      case MulticoreRuntime::TaskType::TASK_GATE_SOFTPLUS:
        route_.Softplus();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_SQRT:
        route_.SqrtScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_ADD_BIAS:
        route_.AddBias();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_TOPK:
        route_.TopKScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GATHER:
        route_.GatherScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_REDUCE_SUM:
        route_.ReduceSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_ADD_EPSILON:
        route_.AddEpsilon();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_DIV:
        route_.DivideSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_MUL_SCALE:
        route_.ScaleSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_CAST_INDEX:
        route_.CastIndices();
        break;
      default:
        Trap();
    }
  }

  __aicore__ inline void ProcessProfiledTile(MulticoreRuntime::CycleTraceRecorder& recorder, uint32_t tile) {
    for (uint32_t stage = 0; stage < kRouteStageCount; ++stage) {
      const uint32_t task_id = this->LoadTaskId(stage);
      const uint64_t start = recorder.Now();
      this->ExecuteTask(task_id);
      recorder.Record(0x30000 + stage + 1, task_id, tile, tile, start, recorder.Now());
    }
  }

  static constexpr uint32_t DIAGNOSTIC_PARAMETER_COUNT = 11;
  __aicore__ inline void PopulateDiagnosticInputList(GM_ADDR* parameters) {
    route_.PopulateDiagnosticInputList(parameters);
  }

 private:
  HyperMegaGate::RoutePipeline route_;
};

}  // namespace
