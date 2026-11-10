/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
namespace gert {
static ge::graphStatus InferShape(InferShapeContext* c) {
  if (c == nullptr || c->GetInputShape(0) == nullptr || c->GetInputShape(1) == nullptr ||
      c->GetAttrs() == nullptr || c->GetInputShape(0)->GetDimNum() != 2 ||
      c->GetInputShape(1)->GetDimNum() != 2) return ge::GRAPH_FAILED;
  auto shape = *c->GetInputShape(0);
  auto t = shape.GetDim(0);
  auto e = c->GetInputShape(1)->GetDim(0);
  for (int i = 0; i < 6; ++i) {
    shape.SetDim(1, (i == 0 || i == 3) ? e : i == 5 ? 1 : 8);
    *c->GetOutputShape(i) = shape;
  }
  const auto tile = *c->GetAttrs()->GetAttrPointer<int64_t>(6);
  if (tile <= 0 || t <= 0) return ge::GRAPH_FAILED;
  const auto profile = *c->GetAttrs()->GetAttrPointer<int64_t>(8);
  const int64_t tiles = (t + tile - 1) / tile;
  const int64_t per_core = (tiles + 31) / 32;
  const int64_t aic = (per_core + 15) / 16 * 16, aiv = (per_core * 11 + 15) / 16 * 16;
  const int64_t bytes = 32 * (64 + 32 * aic) + 64 * (64 + 32 * aiv);
  shape.SetDim(0, 1);
  shape.SetDim(1, profile == 2 ? bytes / 8 : 1);
  *c->GetOutputShape(6) = shape;
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(InferDataTypeContext* c) {
  if (c == nullptr) return ge::GRAPH_FAILED;
  for (int i = 0; i < 7; ++i) c->SetOutputDataType(i, (i == 2 || i == 6) ? ge::DT_INT64 : ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperMegaGateProjection).InferShape(InferShape).InferDataType(InferType);
}  // namespace gert
namespace ops {
class HyperMegaGateProjection : public OpDef {
 public:
  explicit HyperMegaGateProjection(const char* name) : OpDef(name) {
    Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Input("runtime").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    Input("ready").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    Output("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Output("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Output("indices").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Output("selected").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Output("denominator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    Output("trace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    Attr("base_m").AttrType(REQUIRED).Int();
    Attr("base_n").AttrType(REQUIRED).Int();
    Attr("base_k").AttrType(REQUIRED).Int();
    Attr("m_l1").AttrType(REQUIRED).Int();
    Attr("n_l1").AttrType(REQUIRED).Int();
    Attr("k_l1").AttrType(REQUIRED).Int();
    Attr("tile_t").AttrType(REQUIRED).Int();
    Attr("overlap").AttrType(REQUIRED).Int();
    Attr("profile").AttrType(REQUIRED).Int();
    Attr("use_bias").AttrType(REQUIRED).Int();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false);
    AICore().AddConfig("ascend950", config);
  }
};
OP_ADD(HyperMegaGateProjection);
}  // namespace ops
