// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/many_inputs_and_outputs.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string ManyInputsAndOutputs::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ManyInputsAndOutputsParams> obj) {
    std::vector<ov::Shape> inputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (auto i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ManyInputsAndOutputs::SetUp() {
    std::vector<ov::Shape> inputShape;
    std::tie(inputShape, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_shapes_to_test_representation(inputShape));
    auto f = ov::test::snippets::ManyInputsAndOutputsFunction(inputShape);
    function = f.getOriginal();
}

TEST_P(ManyInputsAndOutputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
