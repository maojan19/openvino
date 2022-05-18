// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/max_num_params_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
std::vector<ov::Shape> input_shapes {{1, 64, 10, 10}, {1, 1, 17, 37}, {1, 1, 1, 128}, {1, 1, 1, 3}, {1, 1, 1, 1}, {1, 1, 1, 16}};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, MaxNumParamsEltwiseSinh,
                     ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::Values(12), // 10 Sinh after inputs + Subgraph + Concat
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MaxNumParamsEltwiseSinh::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov