// nnet3/mish-component.h

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

#ifndef KALDI_NNET3_MISH_COMPONENT_H_
#define KALDI_NNET3_MISH_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

class MishComponent: public NonlinearComponent {
 public:
  explicit MishComponent(const MishComponent &other):
      NonlinearComponent(other) { }
  MishComponent() { }
  virtual std::string Type() const { return "MishComponent"; }
  virtual Component* Copy() const { return new MishComponent(*this); }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kPropagateInPlace|
        kStoresStats|(block_dim_ != dim_ ? kInputContiguous : 0);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
 private:
  // this function is called from Backprop code and only does something if the
  // self-repair-scale config value is set.
  void RepairGradients(CuMatrixBase<BaseFloat> *in_deriv,
                       MishComponent *to_update) const;

  MishComponent &operator = (const MishComponent &other); // Disallow.
};

} // namespace nnet3
} // namespace kaldi


#endif
