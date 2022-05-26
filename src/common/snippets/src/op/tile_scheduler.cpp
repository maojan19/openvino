// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile_scheduler.hpp"
#include "snippets/generator.hpp"

ngraph::snippets::op::TileScheduler::TileScheduler(std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> vector_region,
                                                   std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> scalar_region,
                                                   bool is_static)
    : Op(), vector_region{std::move(vector_region)}, scalar_region{std::move(scalar_region)}, is_static{std::move(is_static)} {
}
