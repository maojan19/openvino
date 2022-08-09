// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/convert.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_Convert = {
        { { ov::element::f32 }, { ov::element::i32 } },
        { { ov::element::f32 }, { ov::element::bf16 } },
        { { ov::element::f32 }, { ov::element::u8 } },
        { { ov::element::f32 }, { ov::element::i8 } },

        { { ov::element::bf16 }, { ov::element::f32 } },
        { { ov::element::bf16 }, { ov::element::i32 } },
        { { ov::element::bf16 }, { ov::element::i8 } },
        { { ov::element::bf16 }, { ov::element::u8 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::i32 } },
        { { ov::element::i8 }, { ov::element::bf16 } },
        { { ov::element::i8 }, { ov::element::u8 }  },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::i32 } },
        { { ov::element::u8 }, { ov::element::bf16 } },
        { { ov::element::u8 }, { ov::element::i8 } },
};

const std::vector<std::vector<ov::PartialShape>> inputShapes_Convert = {
        { ov::PartialShape{2, 16} },
        { ov::PartialShape{5, 5} },
        { ov::PartialShape{2, 12, 1} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Convert, Convert,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_Convert),
                                 ::testing::ValuesIn(types_Convert),
                                 ::testing::Values(2),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertInput = {
        { { ov::element::f32 }, { ov::element::i32 } },
        { { ov::element::f32 }, { ov::element::bf16 } },

        { { ov::element::bf16 }, { ov::element::f32 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::i32 } },
        { { ov::element::i8 }, { ov::element::bf16 } },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::i32 } },
        { { ov::element::u8 }, { ov::element::bf16 } },
};

const std::vector<std::vector<ov::PartialShape>> inputShapes_ConvertInput = {
        { ov::PartialShape{2, 16}, ov::PartialShape{1, 16} },
        { ov::PartialShape{5, 18}, ov::PartialShape{5, 1} },
        { ov::PartialShape{3, 1}, ov::PartialShape{3, 21} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertInput, ConvertInput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(3),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertOutput, ConvertOutput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(3),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertStub, ConvertStub,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(4),
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertPartialInputsAndResults = {
        { { ov::element::i8, ov::element::i32, ov::element::f32 }, { ov::element::f32, ov::element::i8 } },
        { { ov::element::bf16, ov::element::u8, ov::element::i32 }, { ov::element::i32, ov::element::bf16 } },
};

const std::vector<std::vector<ov::PartialShape>> inputShapes_ConvertPartialInputsAndResults = {
        { ov::PartialShape{2, 16}, ov::PartialShape{1, 16}, ov::PartialShape{1, 1} },
        { ov::PartialShape{5, 18}, ov::PartialShape{5, 1}, ov::PartialShape{1, 18} },
        { ov::PartialShape{3, 1}, ov::PartialShape{3, 21}, ov::PartialShape{3, 1} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertPartialInputsAndResults, ConvertPartialInputsAndResults,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertPartialInputsAndResults),
                                 ::testing::ValuesIn(types_ConvertPartialInputsAndResults),
                                 ::testing::Values(6),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Convert::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov