// RUN: iree-opt --iree-stablehlo-to-stablehlo-preprocessing \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @batch_norm_inference
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batch_norm_inference(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EPS]], dims = [] : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.broadcast_in_dim %[[STDDEV]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = stablehlo.broadcast_in_dim %[[MEAN]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[X_SCALED:.+]] = stablehlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[X_NORMED:.+]] = stablehlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK:     %[[RESULT:.+]] = stablehlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-NEXT: return %[[RESULT]]
  return %0 : tensor<4x256xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_binary(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
func.func @reorder_broadcast_in_dim_scalar_binary(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[ADD]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[ATAN2:.*]] = stablehlo.atan2 %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[ATAN2]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[DIV:.*]] = stablehlo.divide %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[DIV]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[MAX]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MIN:.*]] = stablehlo.minimum %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[MIN]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[MUL]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[POW:.*]] = stablehlo.power %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[POW]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[REM:.*]] = stablehlo.remainder %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[REM]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SL:.*]] = stablehlo.shift_left %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[SL]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[SRA:.*]] = stablehlo.shift_right_arithmetic %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[SRA]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[SRL:.*]] = stablehlo.shift_right_logical %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[SRL]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[SUB:.*]] = stablehlo.subtract %[[ARG0]], %[[ARG1]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[SUB]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[AND:.*]] = stablehlo.and %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[AND]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[OR:.*]] = stablehlo.or %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[OR]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  // CHECK: %[[XOR:.*]] = stablehlo.xor %[[ARG2]], %[[ARG3]] : tensor<i32>
  // CHECK: stablehlo.broadcast_in_dim %[[XOR]], dims = [] : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %2 = "stablehlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %3 = "stablehlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x8x8x64xi32>
  %4 = stablehlo.add %0, %1 : tensor<1x8x8x64xf32>
  %5 = stablehlo.atan2 %0, %1 : tensor<1x8x8x64xf32>
  %6 = stablehlo.divide %0, %1 : tensor<1x8x8x64xf32>
  %7 = stablehlo.maximum %0, %1 : tensor<1x8x8x64xf32>
  %8 = stablehlo.minimum %0, %1 : tensor<1x8x8x64xf32>
  %9 = stablehlo.multiply %0, %1 : tensor<1x8x8x64xf32>
  %10 = stablehlo.power %0, %1 : tensor<1x8x8x64xf32>
  %11 = stablehlo.remainder %0, %1 : tensor<1x8x8x64xf32>
  %12 = stablehlo.shift_left %2, %3 : tensor<1x8x8x64xi32>
  %13 = stablehlo.shift_right_arithmetic %2, %3 : tensor<1x8x8x64xi32>
  %14 = stablehlo.shift_right_logical %2, %3 : tensor<1x8x8x64xi32>
  %15 = stablehlo.subtract %0, %1 : tensor<1x8x8x64xf32>
  %16 = stablehlo.and %2, %3 : tensor<1x8x8x64xi32>
  %17 = stablehlo.or %2, %3 : tensor<1x8x8x64xi32>
  %18 = stablehlo.xor %2, %3 : tensor<1x8x8x64xi32>
  return %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>, tensor<1x8x8x64xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_binary_diff_type(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>>
func.func @reorder_broadcast_in_dim_scalar_binary_diff_type(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<1x8x8x64xcomplex<f32>> {
  // CHECK: %[[X:.+]] = stablehlo.complex %[[ARG0]], %[[ARG1]] : tensor<complex<f32>>
  // CHECK: stablehlo.broadcast_in_dim %[[X]], dims = [] : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %2 = "stablehlo.complex"(%0, %1) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xcomplex<f32>>
  return %2 : tensor<1x8x8x64xcomplex<f32>>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_binary(%[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func.func @reorder_broadcast_in_dim_1d_binary(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[ATAN2:.*]] = stablehlo.atan2 %[[ARG0]], %[[ARG1]] : tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[ATAN2]], dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %2 = stablehlo.atan2 %0, %1 : tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_2d_binary(%[[ARG0:.*]]: tensor<2x4xi32>, %[[ARG1:.*]]: tensor<2x4xi32>) -> tensor<3x2x4xi32>
func.func @reorder_broadcast_in_dim_2d_binary(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<3x2x4xi32> {
  // CHECK: %[[POWER:.*]] = stablehlo.power %[[ARG0]], %[[ARG1]] : tensor<2x4xi32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[POWER]], dims = [1, 2] : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  %2 = stablehlo.power %0, %1 : tensor<3x2x4xi32>
  // CHECK: return %[[BCAST]]
  return %2 : tensor<3x2x4xi32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary(%[[ARG0:.*]]: tensor<f32>)
func.func @reorder_broadcast_in_dim_scalar_unary(%arg0: tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[ABS]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[CEIL:.*]] = stablehlo.ceil %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[CEIL]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[COSINE:.*]] = stablehlo.cosine %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[COSINE]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[EXP:.*]] = stablehlo.exponential %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[EXP]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[FLOOR:.*]] = stablehlo.floor %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[FLOOR]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[LOG:.*]] = stablehlo.log %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[LOG]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[NEG:.*]] = stablehlo.negate %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[NEG]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[ROUND:.*]] = stablehlo.round_nearest_afz %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[ROUND]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[RSQRT]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SIGN:.*]] = stablehlo.sign %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[SIGN]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SINE:.*]] = stablehlo.sine %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[SINE]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[SQRT:.*]] = stablehlo.sqrt %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] : tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[TANH]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %1 = stablehlo.abs %0 : tensor<1x8x8x64xf32>
  %2 = stablehlo.ceil %0 : tensor<1x8x8x64xf32>
  %3 = stablehlo.cosine %0 : tensor<1x8x8x64xf32>
  %4 = stablehlo.exponential %0 : tensor<1x8x8x64xf32>
  %5 = stablehlo.floor %0 : tensor<1x8x8x64xf32>
  %6 = stablehlo.log %0 : tensor<1x8x8x64xf32>
  %7 = stablehlo.negate %0 : tensor<1x8x8x64xf32>
  %8 = stablehlo.round_nearest_afz %0 : tensor<1x8x8x64xf32>
  %9 = stablehlo.rsqrt %0 : tensor<1x8x8x64xf32>
  %10 = stablehlo.sign %0 : tensor<1x8x8x64xf32>
  %11 = stablehlo.sine %0 : tensor<1x8x8x64xf32>
  %12 = stablehlo.sqrt %0 : tensor<1x8x8x64xf32>
  %13 = stablehlo.tanh %0 : tensor<1x8x8x64xf32>
  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_1d_unary(%[[ARG0:.*]]: tensor<3xf32>) -> tensor<4x3xf32>
func.func @reorder_broadcast_in_dim_1d_unary(%arg0: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[COS:.*]] = stablehlo.cosine %[[ARG0]] : tensor<3xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[COS]], dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  %1 = stablehlo.cosine %0 : tensor<4x3xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<4x3xf32>
}

// -----

// CHECK: @reorder_in_dim_2d_unary(%[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<3x2x4xf32>
func.func @reorder_in_dim_2d_unary(%arg0: tensor<2x4xf32>) -> tensor<3x2x4xf32> {
  // CHECK: %[[LOG:.*]] = stablehlo.log %[[ARG0]] : tensor<2x4xf32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[LOG]], dims = [1, 2] : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  %1 = stablehlo.log %0 : tensor<3x2x4xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<3x2x4xf32>
}

// -----

// CHECK: @reorder_broadcast_in_dim_scalar_unary_diff_type(%[[ARG0:.*]]: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
func.func @reorder_broadcast_in_dim_scalar_unary_diff_type(%arg0: tensor<complex<f32>>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  // CHECK: %[[REAL:.*]] = stablehlo.real %[[ARG0]] : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[REAL]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  // CHECK: %[[IMAG:.*]] = stablehlo.imag %[[ARG0]] : (tensor<complex<f32>>) -> tensor<f32>
  // CHECK: stablehlo.broadcast_in_dim %[[IMAG]], dims = [] : (tensor<f32>) -> tensor<1x8x8x64xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x8x8x64xcomplex<f32>>
  %1 = stablehlo.real %0 : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  %2 = stablehlo.imag %0 : (tensor<1x8x8x64xcomplex<f32>>) -> tensor<1x8x8x64xf32>
  return %1, %2: tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// -----

// CHECK-LABEL: func.func @rng_normal
// CHECK-SAME:              (%[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>)
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x5xf32> {
  %shape = stablehlo.constant dense<[3, 5]> : tensor<2xi64>
  %0 = "stablehlo.rng"(%arg0, %arg1, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %{{.+}} = stablehlo.constant dense<{{.+}}> : tensor<8xf32>
// CHECK:         %[[SIGMA:.+]] = stablehlo.broadcast %[[ARG1]], sizes = [8] : (tensor<f32>) -> tensor<8xf32>
//
//                mag = sigma * sqrt(-2.0 * log(u1)) where sqrt values are
//                constants.
//
// CHECK:         %[[MAG:.+]] = stablehlo.multiply %[[SIGMA]], %{{.+}} : tensor<8xf32>
//
//                z0  = mag * cos(two_pi * u2) + mu;
//                z1  = mag * sin(two_pi * u2) + mu;
//
// CHECK:         %[[MU:.+]] = stablehlo.broadcast %[[ARG0]], sizes = [8] : (tensor<f32>) -> tensor<8xf32>
// CHECK:         %[[T1:.+]] = stablehlo.multiply %[[MAG]], %{{.+}} : tensor<8xf32>
// CHECK:         %[[Z0:.+]] = stablehlo.add %[[T1:.+]], %[[MU]] : tensor<8xf32>
// CHECK:         %[[T2:.+]] = stablehlo.multiply %[[MAG]], %{{.+}} : tensor<8xf32>
// CHECK:         %[[Z1:.+]] = stablehlo.add %[[T2:.+]], %[[MU]] : tensor<8xf32>
//
//                Concate and reshape the output.
// CHECK:         %[[CON:.+]] = stablehlo.concatenate %[[Z0]], %[[Z1]], dim = 0 : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[CON]][0] [15] [1] : tensor<16xf32> to tensor<15xf32>
// CHECK:         %[[RES:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<15xf32>) -> tensor<3x5xf32>
// CHECK:         return %[[RES]]

// -----

// CHECK-LABEL: @mul_float_bool_cast
func.func @mul_float_bool_cast(%arg0 : tensor<?xi1>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
  %1 = "stablehlo.multiply"(%0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[BTOF:.+]] = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
// CHECK: %[[FTOB:.+]] = stablehlo.convert %[[BTOF]] : (tensor<?xf32>) -> tensor<?xi1>
// CHECK: %[[SHP:.+]] = shape.shape_of %[[BTOF]] : tensor<?xf32> -> tensor<1xindex>
// CHECK: %[[BROADCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ZERO]], %[[SHP]], dims = []
// CHECK: %[[SELECT:.+]] = stablehlo.select %[[FTOB]], %arg1, %[[BROADCAST]]

// -----

// CHECK-LABEL: @mul_float_bool_cast_broadcast
func.func @mul_float_bool_cast_broadcast(%arg0: tensor<5xi1>, %arg1: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<5xi1>) -> tensor<5xf32>
  %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5x6xf32>
  %2 = stablehlo.multiply %1, %arg1 : tensor<5x6xf32>
  return %2 : tensor<5x6xf32>
}

// CHECK: stablehlo.select

// -----

// CHECK-LABEL: @mul_float_bool_cast_dyn_broadcast
func.func @mul_float_bool_cast_dyn_broadcast(%arg0: tensor<?xi1>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<?xi1>) -> tensor<?xf32>
    %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
    %2 = "stablehlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %3 = stablehlo.multiply %2, %arg1 : tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
}

// CHECK: stablehlo.select

// -----

// CHECK-LABEL: @dot_general_fuse_both_with_attrs
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<16x64x128xf16>, %[[ARG1:.+]]: tensor<16x128x3072xf16>)
func.func @dot_general_fuse_both_with_attrs(%arg0: tensor<16x64x128xf16>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<16x64x128xf16>) -> tensor<16x64x128xf32>
  %1 = stablehlo.convert %arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: stablehlo.dot_general %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: batching_dims = [0] x [0]
  // CHECK-SAME: contracting_dims = [2] x [1]
  // CHECK-SAME: precision = [DEFAULT, DEFAULT]
  // CHECK-SAME: -> tensor<16x64x3072xf32>
  %2 = "stablehlo.dot_general"(%0, %1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_general_fuse_one
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<{{.+}}xf64>, %[[ARG1:.+]]: tensor<{{.+}}xf16>)
func.func @dot_general_fuse_one(%arg0: tensor<16x64x128xf64>, %arg1: tensor<16x128x3072xf16>) -> tensor<16x64x3072xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<16x64x128xf64>) -> tensor<16x64x128xf32>
  %1 = stablehlo.convert%arg1 : (tensor<16x128x3072xf16>) -> tensor<16x128x3072xf32>
  // CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[ARG0]]
  // CHECK: stablehlo.dot_general %[[CONVERT]], %[[ARG1]]
  %2 = "stablehlo.dot_general"(%0, %1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x128xf32>, tensor<16x128x3072xf32>) -> tensor<16x64x3072xf32>
  return %2 : tensor<16x64x3072xf32>
}

// -----

// CHECK-LABEL: @dot_basic
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<4x4xf16>, %[[ARG1:.+]]: tensor<4x4xf16>)
func.func @dot_basic(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  %1 = stablehlo.convert %arg1 : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK: %[[DOT:.+]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
  %2 = "stablehlo.dot"(%0, %1) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[DOT]]
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @convolution
// CHECK-SAME:     (%[[ARG0:.+]]: tensor<{{.+}}xbf16>, %[[ARG1:.+]]: tensor<{{.+}}xbf16>)
func.func @convolution(%arg0: tensor<16x32x256xbf16>, %arg1: tensor<1x256x256xbf16>) -> tensor<16x32x256xf32> {
  %cast = stablehlo.convert %arg0 : (tensor<16x32x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: -> tensor<16x32x256xf32>
  %0 = "stablehlo.convolution"(%cast, %arg1) {
     batch_group_count = 1 : i64,
     dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
     feature_group_count = 1 : i64,
     lhs_dilation = dense<1> : tensor<1xi64>,
     padding = dense<0> : tensor<1x2xi64>,
     precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
     rhs_dilation = dense<1> : tensor<1xi64>,
     window_strides = dense<1> : tensor<1xi64>
   } : (tensor<16x32x256xf32>, tensor<1x256x256xbf16>) -> tensor<16x32x256xf32>
  // CHECK: return %[[CONV]]
  func.return %0 : tensor<16x32x256xf32>
}

// -----

// CHECK-LABEL: @dynamic_dot_general
// This verifies non-crashing, the lowering to linalg happens elsewhere.
func.func @dynamic_dot_general(%arg1: tensor<?x1024x16x64xf32>, %arg2: tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32> {
  %2 = "stablehlo.dot_general"(%arg2, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 2], rhs_batching_dimensions = [0, 2], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [3]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x1024x16x64xf32>, tensor<?x1024x16x64xf32>) -> tensor<?x16x1024x1024xf32>
  return %2 : tensor<?x16x1024x1024xf32>
}