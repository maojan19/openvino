// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile_scheduler.hpp"
#include "snippets/generator.hpp"

ngraph::snippets::op::TileScheduler::TileScheduler(std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> vector_region,
                                                   std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> scalar_region,
                                                   std::vector<std::vector<size_t>> input_shapes,
                                                   std::vector<std::vector<size_t>> output_shapes,
                                                   std::vector<size_t> exec_domain)
    : Op(), vector_region{std::move(vector_region)}, scalar_region{std::move(scalar_region)},
      input_shapes{std::move(input_shapes)}, output_shapes{std::move(output_shapes)}, exec_domain{std::move(exec_domain)} {
}
