// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::GlobalOptimization {

using IREE::Encoding::EncodingAttr;
using IREE::Encoding::EncodingRole;

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

/// Pads `value` enough for any actual tile sizes that could result from
/// materialization of `encodingAttr`.
bool static isLinalgGenericBroadcast(linalg::GenericOp genericOp) {

  // Check op has single input and output.
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check all iterators are parallel.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
    return false;

  // Check that the two indexing maps are a permutation of each other.
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  bool isTranspose = indexingMaps[1].isIdentity();

  if (!isTranspose)
    return false;

  // Make sure the region only contains a yield op.
  Block &body = genericOp.getRegion().front();
  if (!llvm::hasSingleElement(body))
    return false;

  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

  // The yield op should return the block argument corresponding to the input.
  auto yieldArg = dyn_cast<BlockArgument>(yieldOp.getValues()[0]);
  if (!yieldArg || yieldArg.getArgNumber() != 0 || yieldArg.getOwner() != &body)
    return false;

  return true;
}

static Value pad(OpBuilder &builder, Location loc, Value source,
                 EncodingAttr encodingAttr) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elemType = sourceType.getElementType();
  size_t rank = sourceType.getRank();
  RankedTensorType tensorTypeWithEncoding =
      RankedTensorType::get(sourceType.getShape(), elemType, encodingAttr);
  SmallVector<OpFoldResult> lowPad(rank, builder.getIndexAttr(0));
  SmallVector<Type> resultTypes(rank, builder.getIndexType());

  ValueRange encodingPaddingSizes =
      builder
          .create<IREE::Encoding::UpperBoundTileSizeOp>(
              loc, resultTypes, TypeAttr::get(tensorTypeWithEncoding))
          .getResults();
  SmallVector<OpFoldResult> highPad(rank);
  AffineExpr tileExpr, shapeExpr;
  bindSymbols(builder.getContext(), tileExpr, shapeExpr);
  AffineExpr highPadExpr = shapeExpr.ceilDiv(tileExpr) * tileExpr - shapeExpr;
  for (size_t i = 0; i < rank; ++i) {
    highPad[i] = affine::makeComposedFoldedAffineApply(
        builder, loc, highPadExpr,
        getAsOpFoldResult({encodingPaddingSizes[i],
                           builder.create<tensor::DimOp>(loc, source, i)}));
  }

  Value zero = builder.create<arith::ConstantOp>(loc, elemType,
                                                 builder.getZeroAttr(elemType));
  return builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, source,
                                       lowPad, highPad, zero);
}

Value setEncoding(OpBuilder &builder, Location loc, Value source,
                  EncodingAttr encodingAttr) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto resultType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), encodingAttr);
  return builder.create<IREE::Encoding::SetEncodingOp>(loc, resultType, source);
};

struct MatmulNarrowSizes {
  std::optional<int64_t> M, N;
};

// Returns the minimum of static sizes of the M/N-dimensions in the types of the
// Ouput.
static MatmulNarrowSizes getMatmulNarrowSizes(ShapedType outType,
                                              linalg::LinalgOp linalgOp) {
  linalg::ContractionDimensions cDims =
      linalg::inferContractionDims(linalgOp).value();
  auto map = linalgOp.getIndexingMapsArray().back();
  auto getOutputSizeAtDimPos = [&](unsigned dimPos) -> int64_t {
    return outType.getDimSize(
        map.getResultPosition(getAffineDimExpr(dimPos, linalgOp->getContext()))
            .value());
  };
  // M or N can be empty instead of having an explicit dim size of 1 for matvec
  // and vecmat, so set to 1 if empty.
  int64_t M = cDims.m.empty() ? 1 : getOutputSizeAtDimPos(cDims.m[0]);
  int64_t N = cDims.n.empty() ? 1 : getOutputSizeAtDimPos(cDims.n[0]);

  MatmulNarrowSizes narrow;
  // Threshold below which a M/N size is considered "narrow", making it
  // eligible for a narrow tile size during materialization. This value should
  // be at least as large as the actual M/N tile sizes that we choose on any
  // target in CPUMaterializeEncodingPass. If it is smaller, we will miss
  // opportunities to select optimized narrow tiles for narrow matmuls.
  // If it is larger, everything will work fine, but the IR will be a bit more
  // verbose as more narrow_matmul_{M,N} optional parameters will be specified.
  const int64_t kNarrowThreshold = 16;
  if (!ShapedType::isDynamic(M) && M < kNarrowThreshold) {
    narrow.M = M;
  }
  if (!ShapedType::isDynamic(N) && N < kNarrowThreshold) {
    narrow.N = N;
  }
  return narrow;
}

static Value padBroadcastAndSetEncoding(OpBuilder &builder, Location loc,
                                         linalg::GenericOp broadcastOp, EncodingRole role,
                                         ArrayRef<Type> operandElemTypes,
                                         MatmulNarrowSizes narrow,
                                         ArrayRef<AffineMap> indexingMaps,
                                        Type origType) {
  MLIRContext *ctx = builder.getContext();
  auto encodingForPad = EncodingAttr::get(ctx, role, operandElemTypes,
                                          /*originalType=*/Type{}, narrow.M,
                                          narrow.N, indexingMaps);
  Value source = broadcastOp.getDpsInputs()[0];
  Value padded = pad(builder, loc, source, encodingForPad);

  auto encodingForSetEncoding = encodingForPad;
  if(padded.getType() != source.getType()){
  encodingForSetEncoding = EncodingAttr::get(
      ctx, role, operandElemTypes,
      /*originalType=*/source.getType(), narrow.M, narrow.N, indexingMaps);
  }

  Value encodedInput =  setEncoding(builder, loc, padded, encodingForSetEncoding);

  Value updatedGeneric = builder.create<linalg::GenericOp>(
      loc, broadcastOp.getResultTypes(), ValueRange{encodedInput},
      broadcastOp.getDpsInits(), broadcastOp.getIndexingMapsArray(),
      broadcastOp.getIteratorTypesArray(),
      [=](OpBuilder &b, Location loc, ValueRange args) {
        Value value = args[0];
        b.create<linalg::YieldOp>(loc, value);
      }).getResult(0);

  return updatedGeneric;
}

static Value padAndSetEncoding(OpBuilder &builder, Location loc, Value source,
                               EncodingRole role,
                               ArrayRef<Type> operandElemTypes,
                               MatmulNarrowSizes narrow,
                               ArrayRef<AffineMap> indexingMaps) {
  MLIRContext *ctx = builder.getContext();

  // auto broadcastOp =  source.getDefiningOp<linalg::GenericOp>();
  // if (broadcastOp)
  //   return padBroadcastAndSetEncoding(builder, loc, broadcastOp, role,
  //                                   operandElemTypes, narrow, indexingMaps,
  //                                     source.getType());


  // No need to specify original_type in the encoding poadded to pad(), because
  // the operand there is the `source` tensor, so it will default to reading its
  // original shape.
  auto encodingForPad = EncodingAttr::get(ctx, role, operandElemTypes,
                                          /*originalType=*/Type{}, narrow.M,
                                          narrow.N, indexingMaps);
  Value padded = pad(builder, loc, source, encodingForPad);
  // For setEncoding() below, we potentially need to specify an encoding with an
  // explicit original_type, because the operand there is the padded tensor
  // returned by pad() above, but we want setEncoding to be aware of the
  // original source tensor shape, not the padded tensor shape. To limit IR
  // verbosity, we only specify the original original_type when it differs from
  // the tensor type that the encoding is applied to.
  auto encodingForSetEncoding = encodingForPad;
  if (padded.getType() != source.getType()) {
    encodingForSetEncoding = EncodingAttr::get(
        ctx, role, operandElemTypes,
        /*originalType=*/source.getType(), narrow.M, narrow.N, indexingMaps);
  }
  return setEncoding(builder, loc, padded, encodingForSetEncoding);
}

static Value unsetEncodingAndExtractSlice(OpBuilder &builder, Location loc,
                                          Value source,
                                          SmallVector<OpFoldResult> sizes) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  auto unsetEncoding = builder
                           .create<IREE::Encoding::UnsetEncodingOp>(
                               loc, unsetEncodingReturnType, source)
                           .getResult();
  auto rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, unsetEncoding, offsets,
                                                sizes, strides);
}

/// Given a LinalgOp and one of its OpOperands, return the element type,
/// inferring unsignedness from the body of the LinalgOp
static Type getContractionInputTypeWithSignedness(OpBuilder &builder,
                                                  linalg::LinalgOp linalgOp,
                                                  OpOperand *operand) {
  assert(linalg::isaContractionOpInterface(linalgOp));
  assert(operand->getOwner() == linalgOp.getOperation());
  auto elemType = getElementTypeOrSelf(operand->get().getType());
  // Infer if unsigned from body ops
  Value blockArg = linalgOp.getMatchingBlockArgument(operand);
  for (auto bodyCastOp : blockArg.getParentBlock()->getOps<arith::ExtUIOp>()) {
    if (bodyCastOp->getOperand(0) == blockArg) {
      return builder.getIntegerType(elemType.getIntOrFloatBitWidth(),
                                    /*isSigned=*/false);
    }
  }
  return elemType;
}

/// Returns true iff the linalgOp has a body like a regular matmul, i.e.
/// yield(add(out, mul(cast(in0), cast(in1))))
static bool hasMatmulLikeBody(linalg::LinalgOp linalgOp) {
  auto outBlockArg =
      linalgOp.getMatchingBlockArgument(linalgOp.getDpsInitOperand(0));
  auto yieldOp =
      dyn_cast<linalg::YieldOp>(outBlockArg.getParentBlock()->getTerminator());
  if (!yieldOp) {
    return false;
  }
  auto addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  auto addLhs = addOp->getOperand(0);
  auto addRhs = addOp->getOperand(1);
  auto addLhsOp = addLhs.getDefiningOp();
  auto addRhsOp = addRhs.getDefiningOp();
  if (!(addLhsOp && addRhs == outBlockArg) &&
      !(addRhsOp && addLhs == outBlockArg)) {
    return false;
  }
  Operation *mulOp = addLhsOp ? addLhsOp : addRhsOp;
  if (!isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }
  auto mulLhs = mulOp->getOperand(0);
  auto mulRhs = mulOp->getOperand(1);
  auto mulLhsOp = mulLhs.getDefiningOp<CastOpInterface>();
  auto mulRhsOp = mulRhs.getDefiningOp<CastOpInterface>();
  if (!isa<BlockArgument>(mulLhs) && !mulLhsOp && !isa<BlockArgument>(mulRhs) &&
      !mulRhsOp) {
    return false;
  }
  if ((mulLhsOp && !isa<BlockArgument>(mulLhsOp->getOperand(0))) ||
      (mulRhsOp && !isa<BlockArgument>(mulRhsOp->getOperand(0)))) {
    return false;
  }
  return true;
}

/// Not all contractions are supported by data tiling, so return true if:
///   1) linalgOp has contraction indexingMaps.
///   2) There are not more than one of each contraction dimension
///   3) There is and M or N dimension, and there is a K dimension
///   4) linalgOp has the same body as an ordinary int or float matmul
///
/// These restrictions are required because data tiling currently creates
/// an Mmt4DOp or BatchMmt4DOp on the packed inputs.
///
/// TODO(#16176): Loosen restrictions on contraction ops once data tiling
/// can support more cases.
static LogicalResult isSupportedContractionOp(PatternRewriter &rewriter,
                                              linalg::LinalgOp linalgOp) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expected isaContractionOpInterface");
  }
  auto cDims = linalg::inferContractionDims(linalgOp);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected {|Batch|, |M|, |N|, |K|} <= 1");
  }
  if ((cDims->n.empty() && cDims->m.empty()) || cDims->k.empty()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected M or N dims and K dim to not be empty");
  }
  if (!hasMatmulLikeBody(linalgOp)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected op to have a matmul body, i.e. yield(add(out, "
                  "mul(cast(in0), cast(in1))))");
  }
  return success();
}

namespace {

class setContractionOpEncoding
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  explicit setContractionOpEncoding(MLIRContext *ctx, int64_t factor)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), padFactor(factor) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasPureTensorSemantics()) {
      return failure();
    }
    if (getCompilationInfo(linalgOp)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the op has preset compilation strategy, skip SetEncoding");
    }
    if (failed(isSupportedContractionOp(rewriter, linalgOp))) {
      return failure();
    }

    auto inputs = linalgOp.getDpsInputs();
    auto outputs = linalgOp.getDpsInits();

    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value out = outputs[0];

    Type lhsElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInputOperand(0));
    Type rhsElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInputOperand(1));
    Type outElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInitOperand(0));

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }
    SmallVector<Type> elemTypes = {lhsElemType, rhsElemType, outElemType};

    MatmulNarrowSizes narrowSizes =
        getMatmulNarrowSizes(cast<ShapedType>(out.getType()), linalgOp);

    Location loc = linalgOp.getLoc();
    SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
    Value encodedLhs, encodedRhs, encodedOut;

    if (!padFactor) {
      encodedLhs = padAndSetEncoding(rewriter, loc, lhs, EncodingRole::LHS,
                                     elemTypes, narrowSizes, maps);
      encodedRhs = padAndSetEncoding(rewriter, loc, rhs, EncodingRole::RHS,
                                     elemTypes, narrowSizes, maps);
      encodedOut = padAndSetEncoding(rewriter, loc, out, EncodingRole::RESULT,
                                     elemTypes, narrowSizes, maps);
    } else {
      auto setEncodingWrapper = [&](Value src, EncodingRole role) -> Value {
        SmallVector<int64_t> roundDimsTo(linalgOp.getNumLoops(), padFactor);
        auto encoding = EncodingAttr::get(
            linalgOp.getContext(), role, elemTypes, src.getType(),
            narrowSizes.M, narrowSizes.N, maps, roundDimsTo);
        return setEncoding(rewriter, loc, src, encoding);
      };
      encodedLhs = setEncodingWrapper(lhs, EncodingRole::LHS);
      encodedRhs = setEncodingWrapper(rhs, EncodingRole::RHS);
      encodedOut = setEncodingWrapper(out, EncodingRole::RESULT);
    }
    Value opTiled = clone(rewriter, linalgOp, encodedOut.getType(),
                          ValueRange{encodedLhs, encodedRhs, encodedOut})
                        ->getResult(0);

    // Sizes are computed by original output size.
    FailureOr<SmallVector<OpFoldResult>> outSizes =
        IREE::LinalgExt::getDims(rewriter, loc, out);
    if (failed(outSizes)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to get shape of result");
    }

    Value result =
        unsetEncodingAndExtractSlice(rewriter, loc, opTiled, outSizes.value());

    rewriter.replaceOp(linalgOp, result);
    return success();
  }

private:
  int64_t padFactor = 0;
};

/// Pattern to fold a `linalg.fill` -> `iree_encoding.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::getMixedSizes(rewriter, loc, fillOp.getOutputs()[0]);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(encodingOp, fillOp.getInputs(),
                                                ValueRange{newEmptyOp});
    return success();
  }
};


/// Pattern to fold a `linalg.broadcast` -> `iree_encoding.set_encoding`
/// operation into a `linalg.broadcast` of the encoded type.
struct FoldBroadcastWithSetEncoding
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {

    auto broadcastOp = encodingOp.getSource()
                           .getDefiningOp<tensor::PadOp>()
                           .getOperand(0)
                           .getDefiningOp<linalg::GenericOp>();
    if (!broadcastOp)
      return failure();

    if (!isLinalgGenericBroadcast(broadcastOp))
      return failure();

    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    SmallVector<Type> newResultTypes;
    SmallVector<AffineMap> maps;
    for (OpOperand *in : broadcastOp.getDpsInputOperands()) {
        newInputs.push_back(in->get());
    }
    for (OpOperand &out : broadcastOp.getDpsInitsMutable()) {
        newOutputs.push_back(out.get());
        newResultTypes.push_back(out.get().getType());
    }

    RankedTensorType encodingType = encodingOp.getResultType();
    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    Location loc = broadcastOp.getLoc();

    SmallVector<OpFoldResult> dimValues = tensor::getMixedSizes(
        rewriter, loc, broadcastOp.getDpsInputs()[0]);

    SmallVector<OpFoldResult> dimOutValues = tensor::getMixedSizes(
        rewriter, loc, broadcastOp.getDpsInits()[0]);

    EncodingAttr encoding = IREE::Encoding::getEncodingAttr(encodingType);

    encoding.dump();

    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());


    auto copyOp = rewriter.create<linalg::CopyOp>(loc, newInputs,
                                                ValueRange{newEmptyOp}).getResult(0);

    auto outtType = dyn_cast<RankedTensorType>(newOutputs[0].getType());
    auto encodedOutType = RankedTensorType::get(
      outtType.getShape(), outtType.getElementType(),
      encodingType.getEncoding());

    auto newemptyout = rewriter.create<tensor::EmptyOp>(
        loc, dimOutValues, encodedOutType.getElementType(),
        encodedOutType.getEncoding());

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        encodingOp, encodedOutType, ValueRange{copyOp}, ValueRange{newemptyout},
        broadcastOp.getIndexingMapsArray(), broadcastOp.getIteratorTypesArray(),
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value value = args[0];
          b.create<linalg::YieldOp>(loc, value);
        });

    return success();
  }
};

struct SetEncodingPass : public SetEncodingBase<SetEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Encoding::IREEEncodingDialect>();
  }
  explicit SetEncodingPass(int64_t factor) { this->padFactor.setValue(factor); }

  void runOnOperation() override;
};
} // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<setContractionOpEncoding>(context, padFactor);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<FoldFillWithSetEncoding>(context);
    // patterns.insert<FoldBroadcastWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSetEncodingPass(int64_t padFactor) {
  return std::make_unique<SetEncodingPass>(padFactor);
}

} // namespace mlir::iree_compiler::GlobalOptimization
