// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Tile
 * @brief Generated by Canonicalization and represents Loop in affine notation
 * @ingroup snippets
 */
class Tile : public ngraph::op::Op {
public:
    OPENVINO_OP("Tile", "SnippetsOpset");

    Tile(const std::vector<AllocatedEmitter>& region);
    Tile() = default;
    std::vector<AllocatedEmitter> region;
    size_t increment = 0;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    std::vector<size_t> io_dims {};

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<Tile>(region);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph