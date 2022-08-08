// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "./snippets_helpers.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
/// The most trivial graph, just one Convert.
/// Tokenized simply by starting subgraph.
//    in1
//  Convert
//   Result
class ConvertFunction : public SnippetsFunctionBase {
public:
    explicit ConvertFunction(const std::vector<PartialShape>& inputShapes,
                             const ov::element::Type inType = ov::element::f32,
                             const ov::element::Type outType = ov::element::u8)
    : SnippetsFunctionBase(inputShapes), inType(inType), outType(outType) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    ov::element::Type inType;
    ov::element::Type outType;
};


/// The one of the input of Add is Convert
/// Tokenized simply by starting subgraph.
//    in1
//  Convert    in2
//       Add
//      Result
class ConvertInputFunction : public SnippetsFunctionBase {
public:
    explicit ConvertInputFunction(const std::vector<PartialShape>& inputShapes,
                                  const ov::element::Type inType = ov::element::f32,
                                  const ov::element::Type outType = ov::element::u8)
    : SnippetsFunctionBase(inputShapes), inType(inType), outType(outType) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    ov::element::Type inType;
    ov::element::Type outType;
};

/// The output of Sub is Convert
/// Tokenized simply by starting subgraph.
//    in1     in2
//       Sub
//     Convert
//      Result
class ConvertOutputFunction : public SnippetsFunctionBase {
public:
    explicit ConvertOutputFunction(const std::vector<PartialShape>& inputShapes,
                                   const ov::element::Type inType = ov::element::f32,
                                   const ov::element::Type outType = ov::element::i8)
    : SnippetsFunctionBase(inputShapes), inType(inType), outType(outType) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    ov::element::Type inType;
    ov::element::Type outType;
};


/// There are 2 subgraphs: Add + Convert(Stub) and Relu
/// Tokenized simply by starting subgraph.
//    in1    in2           in1     in2
//       Add                 Subgraph
//     Convert        ->        |
//       Relu                Subgraph
//      Result                Result
class ConvertStubFunction : public SnippetsFunctionBase {
public:
    explicit ConvertStubFunction(const std::vector<PartialShape>& inputShapes,
                                 const ov::element::Type inType = ov::element::f32,
                                 const ov::element::Type outType = ov::element::i8)
        : SnippetsFunctionBase(inputShapes), inType(inType), outType(outType) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    ov::element::Type inType;
    ov::element::Type outType;
};


/// Not all Inputs and Results have Convert
/// Tokenized simply by starting subgraph.
//    in1      in2
//  Convert  Convert
//        Add
//       Relu        in3
//  Convert     Sub
//  Result1  Unsqueeze   <- It's to avoid many result output for subgraph (it's a limitation of collapsing)
//            Result2
class ConvertPartialInputsAndResultsFunction : public SnippetsFunctionBase {
public:
    explicit ConvertPartialInputsAndResultsFunction(const std::vector<PartialShape>& inputShapes,
                                                    const std::vector<ov::element::Type>& inTypes = {ov::element::f32},
                                                    const std::vector<ov::element::Type>& outTypes = {ov::element::f32})
    : SnippetsFunctionBase(inputShapes), inTypes(inTypes), outTypes(outTypes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    std::vector<ov::element::Type> inTypes;
    std::vector<ov::element::Type> outTypes;
};


}  // namespace snippets
}  // namespace test
}  // namespace ov
