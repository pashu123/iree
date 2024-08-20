// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUGENERICBROADCASTTOEXPANDSHAPEPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct GenericToExpandShape final : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    auto loc = genericOp.getLoc();
    // get the output tensor
    auto inputTensor = genericOp.getInputs()[0];
    auto outputTensor = genericOp.getOutputs()[0];
    auto outputType = cast<RankedTensorType>(outputTensor.getType());
    auto inputSizes = cast<RankedTensorType>(inputTensor.getType()).getShape();
    auto outputSizes =
        cast<RankedTensorType>(outputTensor.getType()).getShape();

    if (inputSizes.size() == outputSizes.size()) {
      llvm::errs() << "Input and output sizes are the same\n";
      return failure();
    }

    llvm::errs() << "Ouput sizes: " << outputSizes.size();
    llvm::errs() << "Input sizes: " << inputSizes.size();

    SmallVector<ReassociationIndices> reassociationIdx(inputSizes.size());

    for (auto i = 2; i <= inputSizes.size(); i++) {
      reassociationIdx[i - 1].push_back(i);
    }

    reassociationIdx.front().push_back(0);
    reassociationIdx.front().push_back(1);

    outputTensor.dump();

    auto expandShapeOp = rewriter
                             .create<tensor::ExpandShapeOp>(
                                 loc, outputType, inputTensor, reassociationIdx)
                             .getResult();
    expandShapeOp.dump();
    rewriter.replaceOp(genericOp, expandShapeOp);
    return success();
  }
};

struct LLVMCPUGenericBroadCastToExpandShapePass
    : public impl::LLVMCPUGenericBroadCastToExpandShapePassBase<
          LLVMCPUGenericBroadCastToExpandShapePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    RewritePatternSet broadCastToExpandShapePatterns(&getContext());
    broadCastToExpandShapePatterns.insert<GenericToExpandShape>(
        fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(
            fn, std::move(broadCastToExpandShapePatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
