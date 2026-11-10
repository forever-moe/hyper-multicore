/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/library.h>

#include "mega_gate_op_api.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
// In-tree scheduler ABI: allocate_runtime_config(11, num_workers=64), including scratch.
constexpr int64_t kRuntimeBytes = 34240;

void CheckProjectionShape(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "Expected [T,H] and [E,H]");
  TORCH_CHECK(x.size(0) > 0 && x.size(0) <= 0x7fffffff && x.size(1) > 0 && x.size(1) <= 0x7fffffff &&
                  x.size(1) % 64 == 0 && w.size(0) >= 8 && w.size(0) <= 384,
              "Unsupported positive T/H/E");
}

void CheckProjectionProperties(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.device() == w.device(), "Same NPU required");
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && x.is_contiguous() && w.is_contiguous(),
              "Contiguous FP32 projection required");
  TORCH_CHECK(!at::GradMode::is_enabled() || (!x.requires_grad() && !w.requires_grad()),
              "Use sigmoid_route_from_hidden_states for training");
}

void CheckProjectionResources(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias,
                              const at::Tensor& runtime) {
  TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == at::kFloat && bias.dim() == 1 &&
                  bias.numel() == w.size(0) && bias.is_contiguous() && !bias.requires_grad(),
              "Invalid expert bias");
  TORCH_CHECK(runtime.device() == x.device() && runtime.scalar_type() == at::kByte && runtime.dim() == 1 &&
                  runtime.is_contiguous() && runtime.numel() == kRuntimeBytes,
              "Invalid pinned runtime ABI");
}

at::Tensor ProjectionTrace(const at::Tensor& x, const c10::optional<at::Tensor>& buffer) {
  if (!buffer.has_value()) return at::empty({1, 1}, x.options().dtype(at::kLong));
  const int64_t tiles = (x.size(0) + 31) / 32;
  const int64_t per_core = (tiles + 31) / 32;
  const int64_t aic_capacity = (per_core + 15) / 16 * 16;
  const int64_t aiv_capacity = (11 * per_core + 15) / 16 * 16;
  const int64_t bytes = 32 * (64 + 32 * aic_capacity) + 64 * (64 + 32 * aiv_capacity);
  const auto& value = buffer.value();
  TORCH_CHECK(value.device() == x.device() && value.scalar_type() == at::kByte && value.dim() == 1 &&
                  value.is_contiguous() && value.numel() == bytes,
              "Invalid cycle diagnostic buffer");
  return value.view(at::kLong).view({1, bytes / 8});
}

Outputs Projection(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias, const at::Tensor& runtime,
                   const c10::optional<at::Tensor>& diagnostic_buffer) {
  CheckProjectionShape(x, w);
  CheckProjectionProperties(x, w);
  const c10_npu::NPUGuard device_guard(x.device());
  CheckProjectionResources(x, w, bias, runtime);
  int64_t bm = 128, bn = 128, bk = 64, ml = 128, nl = 128, kl = 256;
  int64_t tile = 32, overlap = 1, use_bias = 1;
  int64_t profile = diagnostic_buffer.has_value() ? 2 : 0;
  auto trace = ProjectionTrace(x, diagnostic_buffer);
  const int64_t t = x.size(0), e = w.size(0), tiles = (t + tile - 1) / tile;
  auto logits = at::empty({t, e}, x.options());
  auto weights = at::empty({t, 8}, x.options());
  auto indices = at::empty({t, 8}, x.options().dtype(at::kLong));
  auto scores = at::empty({t, e}, x.options());
  auto selected = at::empty({t, 8}, x.options());
  auto denominator = at::empty({t, 1}, x.options());
  auto ready = at::zeros({tiles * 32}, x.options().dtype(at::kInt));
  static const hyper_multicore::mega_gate::CachedOpApi api("aclnnHyperMegaGateProjection",
                                                          "aclnnHyperMegaGateProjectionGetWorkspaceSize");
  hyper_multicore::mega_gate::execute_cached_op(api, x, w, bias, runtime, ready, bm, bn, bk, ml, nl, kl, tile, overlap,
                                               profile, use_bias, logits, weights, indices, scores, selected,
                                               denominator, trace);
  return {logits, weights, indices, scores, selected, denominator, trace};
}

at::Tensor ProjectionMm(const at::Tensor& left, const at::Tensor& right) {
  TORCH_CHECK(left.dim() == 2 && right.dim() == 2 && left.size(1) == right.size(0),
              "Projection derivative requires matching two-dimensional matrices");
  TORCH_CHECK(left.device().type() == c10::DeviceType::PrivateUse1 && left.device() == right.device() &&
                  left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat,
              "Projection derivative requires FP32 matrices on the same NPU");
  const c10_npu::NPUGuard device_guard(left.device());
  auto output = at::empty({left.size(0), right.size(1)}, left.options());
  int8_t cube_math_type = 0;  // KEEP_DTYPE is independent of the global HF32 policy.
  static const hyper_multicore::mega_gate::CachedOpApi api("aclnnMatmul", "aclnnMatmulGetWorkspaceSize");
  hyper_multicore::mega_gate::execute_cached_op(api, left, right, output, cube_math_type);
  return output;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def(
      "mega_gate_projection(Tensor x, Tensor weight, Tensor bias, Tensor runtime, "
      "Tensor? diagnostic_buffer=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("mega_gate_projection_mm(Tensor left, Tensor right) -> Tensor");
}
TORCH_LIBRARY_IMPL(hyper_multicore, PrivateUse1, m) {
  m.impl("mega_gate_projection", &Projection);
  m.impl("mega_gate_projection_mm", &ProjectionMm);
}
