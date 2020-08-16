// nnet3/mish-component.cc

// Copyright 2020  AlexHT Hung

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/mish-component.h"

#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

void *MishComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const {
  CuMatrix<BaseFloat> tmp(in.NumRows(), in.NumCols(), kUndefined);
  tmp.SoftHinge(in);
  out->Tanh(tmp);
  out->MulElements(in);
  return NULL;
}

void DiffMish(const CuMatrixBase<BaseFloat> &in_value,
              CuMatrixBase<BaseFloat> *in_deriv) {
  CuMatrix<BaseFloat> x_sigmoid(in_value.NumRows(), in_value.NumCols(),
                                kUndefined);
  CuMatrix<BaseFloat> x_sp(in_value.NumRows(), in_value.NumCols(), kUndefined);
  CuMatrix<BaseFloat> x_tanh_sp(in_value.NumRows(), in_value.NumCols(),
                                kUndefined);
  x_sigmoid.Sigmoid(in_value);
  x_sp.SoftHinge(in_value);
  x_tanh_sp.Tanh(x_sp);
  in_deriv->DiffTanh(x_tanh_sp, x_sigmoid);
  in_deriv->MulElements(in_value);
  in_deriv->AddMat(1.0, x_tanh_sp);
}

void MishComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo, Component *to_update_in,
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("MishComponent::Backprop");
  if (in_deriv != NULL) {
    DiffMish(in_value, in_deriv);
    in_deriv->MulElements(out_deriv);

    MishComponent *to_update = dynamic_cast<MishComponent *>(to_update_in);
    if (to_update != NULL) {
      RepairGradients(in_deriv, to_update);
      to_update->StoreBackpropStats(out_deriv);
    }
  }
}

/*
  Note on the derivative of the softplus function:
  softplus'(x) = sigmoid(x)
*/
void MishComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value,
                               void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0) return;
  // derivative of the onlinearity is out_value * (1.0 - out_value);
  CuMatrix<BaseFloat> temp_deriv(in_value.NumRows(), in_value.NumCols(),
                                 kUndefined);
  DiffMish(in_value, &temp_deriv);
  StoreStatsInternal(out_value, &temp_deriv);
}

void MishComponent::RepairGradients(CuMatrixBase<BaseFloat> *in_deriv,
                                    MishComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  int32 dim = dim_, block_dim = block_dim_;
  BaseFloat default_lower_threshold = 0.05, default_upper_threshold = 0.95;
  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;
  KALDI_ASSERT(in_deriv->NumCols() == dim || in_deriv->NumCols() == block_dim);
  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim)
    return;

  if (in_deriv->NumCols() != block_dim) {
    KALDI_ASSERT(in_deriv->NumCols() == in_deriv->Stride());
    int32 dim_multiple = dim / block_dim;
    CuSubMatrix<BaseFloat> in_deriv_reshaped(in_deriv->Data(),
                                             in_deriv->NumRows() * dim_multiple,
                                             block_dim, block_dim);
    RepairGradients(&in_deriv_reshaped, to_update);
    return;
  }

  // By now we know that in_deriv->NumCols() == block_dim.

  if (RandUniform() > repair_probability) return;

  to_update->num_dims_processed_ += block_dim;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold;  // -1000.0
  BaseFloat count = count_,
            lower_threshold = (self_repair_lower_threshold_ == unset
                                   ? default_lower_threshold
                                   : self_repair_lower_threshold_) *
                              count,
            upper_threshold = (self_repair_upper_threshold_ == unset
                                   ? default_upper_threshold
                                   : self_repair_upper_threshold_) *
                              count;

  CuMatrix<BaseFloat> storage(2, block_dim + 2, kUndefined);
  CuSubVector<BaseFloat> thresholds_vec(storage.RowData(0) + block_dim, 2);
  CuSubMatrix<BaseFloat> stats_mat(storage, 0, 2, 0, block_dim);
  thresholds_vec(0) = -lower_threshold;
  thresholds_vec(1) = -upper_threshold;
  CuSubVector<BaseFloat> row0(stats_mat, 0);
  CuSubVector<BaseFloat> row1(stats_mat, 1);

  if (block_dim == dim) {
    row0.CopyFromVec(deriv_sum_);
  } else {
    CuSubMatrix<double> deriv_sum_mat(deriv_sum_.Data(), dim / block_dim,
                                      block_dim, block_dim);
    CuVector<double> deriv_sum_dbl(block_dim);
    // get the average of the deriv-sums over the blocks.
    deriv_sum_dbl.AddRowSumMat(block_dim * 1.0 / dim, deriv_sum_mat);
    row0.CopyFromVec(deriv_sum_dbl);
  }
  row1.CopyFromVec(row0);
  stats_mat.AddVecToCols(1.0, thresholds_vec, 1.0);
  // now row0 equals stats - lower_threshold, and
  //     row1 equals stats - upper_threshold.
  stats_mat.ApplyHeaviside();
  // now row0 equals (stats > lower_threshold ? 1 : 0), and
  //     row1 equals (stats > upper_threshold ? 1 : 0).
  // what we want is:
  // self_repair_scale * ((stats <= lower_threshold ? 1 : 0) +
  //                         (stats > upper_threshold ? -1 : 0)).
  //
  // we can get these in stats_mat.Row(0) by computing:
  // -self_repair_scale * (stats_mat.Row(1)  + stats_mat.Row(0) - 1).
  row0.AddVec(1.0, row1, 1.0);
  row0.Add(-1.0);
  CuVector<BaseFloat> temp(row0);
  temp.ApplyPow(2.0);
  to_update->num_dims_self_repaired_ += temp.Sum();
  // [actually we need to divide by repair_probability also, to
  //  correct for the fact that we only do this on some frames.]
  row0.Scale(-self_repair_scale_ / repair_probability);
  in_deriv->AddVecToRows(1.0, row0, 1.0);
}

}  // namespace nnet3
}  // namespace kaldi
