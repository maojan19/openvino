// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>
#include <tuple>

#include <dnnl_debug.h>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include <snippets/op/subgraph.hpp>
#include "emitters/cpu_generator.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

Snippet::Snippet(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_common) ?
        dnnl::impl::cpu::x64::avx512_common : dnnl::impl::cpu::x64::avx2;

    // Create a deep local copy of the input snippet to perform canonicalization & code generation
    // Todo: Probably better to implement a proper copy constructor
    if (const auto tmp_snippet =  ov::as_type_ptr<ngraph::snippets::op::Subgraph>(op)) {
        ngraph::OutputVector subgraph_node_inputs;
        for (const auto &input : tmp_snippet->input_values()) {
            auto new_input = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            subgraph_node_inputs.push_back(new_input);
        }
        auto new_body = ov::clone_model(*tmp_snippet->get_body().get());
        snippet = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_node_inputs, new_body);
        ngraph::copy_runtime_info(tmp_snippet, snippet);
        snippet->set_friendly_name(tmp_snippet->get_friendly_name());
        snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
    } else {
        IE_THROW(NotImplemented) << "Node is not an instance of snippets::op::Subgraph";
    }
}

void Snippet::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const Precision supportedPrecision = Precision::FP32;

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1, 2, 4, 5) && dimRanksAreEqual;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    const bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4, 5) && dimRanksAreEqual;
    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };
    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, Precision prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = mayiuse(dnnl::impl::cpu::x64::avx512_common) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        size_t offset = 0;
        NodeConfig config;
        config.dynBatchSupport = false;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            BlockedMemoryDesc::CmpMask inputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace()) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(inputShapes[i], supportedPrecision, offset), inputMask);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            BlockedMemoryDesc::CmpMask outputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(outputShapes[i], supportedPrecision, offset), outputMask);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (mayiuse(x64::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Snippet::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}
void Snippet::calcJITParams(std::vector<size_t>& offsets, std::vector<int64_t>& sch_offsets) {
    const auto& static_master_shape = master_shape.get_shape();
    const size_t numInputs = bodyInputShapes.size();
    const size_t numParams = numInputs + bodyOutputShapes.size();
    // Note that wen don't need offset for the last dim, since it's handled directly by Load/Store emitters
    const size_t offset_rank = master_shape.size() - 1;
    offsets.resize(numParams * (offset_rank), 1);
    auto offset_calculation = [this, offset_rank, static_master_shape](size_t *off, const std::vector<size_t>& dims) {
        size_t k = dims.back();
        for (int i = offset_rank - 1; i >= 0; i--) {
            auto tmp = (dims[i] == static_master_shape[i]) ? k : 0;
            off[i] = tmp;
            k *= dims[i];
        }
    };
    for (size_t i = 0; i < numParams; i++) {
        offset_calculation(offsets.data() + i * offset_rank, i < numInputs ? bodyInputShapes[i].get_shape() : bodyOutputShapes[i - numInputs].get_shape());
    }
    for (auto &d : offsets)
        d *= dataSize;

    sch_offsets = std::vector<int64_t>(numParams, 0);
    if (tileRank > 1) {
        // todo: simplify pointer increment logics. Currently some increments are performed by emitters
        //  (not always, but on condition), and some - by TileScheduler.
        // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
        for (size_t i = 0; i < bodyInputShapes.size(); i++) {
            // the last offset is ignored, so offsets[offset_rank - 1] is actually outer tile offset
            int64_t off = offsets[(i + 1) * offset_rank - 1];
            const auto& input_shape = bodyInputShapes[i].get_shape();
            if (off > dataSize) {
                sch_offsets[i] = 0;
                // offset == data_size only if input_shape.back() == 1, but ScalarLoadEmitter doesn't perform increment
                // in such cases, because it thinks it's broadcasting.
            } else if (off == dataSize) {
                sch_offsets[i] = off;
                // if outer tile is broadcasted then we need to step back to read the same data once again
            } else if (input_shape[master_shape.size() - 2] != static_master_shape[master_shape.size() - 2] && input_shape.back() != 1) {
//                sch_offsets[i] = -1 * master_shape.back() * dataSize;
                sch_offsets[i] = -1 * static_master_shape[master_shape.size() - 1] * dataSize;
            }
        }
        // we need to step back for outputs too if output shape is not equal to master_shape
        for (size_t i = 0; i < bodyOutputShapes.size(); i++) {
            int64_t off = offsets[(i + 1 + numInputs) * offset_rank - 1];
            sch_offsets[i + numInputs] = off - static_master_shape.back() * dataSize;
        }
    }
}
void Snippet::optimizeExecDomain() {
    auto findDimsToCollapse = [this](PartialShape &domain, size_t workAmount) {
        auto collapseLastDims = [](PartialShape& dims, size_t dimsToCollapse) {
            if (dimsToCollapse >= dims.size() - 1)
                IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
            for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 1;
            }
        };
        int collapsedDims = 0;
        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = domain[domain.size() - 1].get_length();
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < workAmount) {
            if (static_cast<int>(domain.size()) - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (size_t i = 0; i < bodyInputShapes.size(); i++) {
                const size_t last = bodyInputShapes[i].size() - 1;
                if ((bodyInputShapes[i][last - 1] != 1 && bodyInputShapes[i][last] == 1) ||
                    (bodyInputShapes[i][last - 1] == 1 && bodyInputShapes[i][last] != 1)) {
                    canCollapse = false;
                    break;
                }
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * domain[domain.size() - 2].get_length();
            if (workAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                // if we cannot use dim collapsing we should use tile2D
                if (!canCollapse) {
                    if (tileRank < maxTileRank) {
                        tileRank++;
                        continue;
                    }

                    break;
                }
                collapsedDims++;
                for (auto &d : bodyInputShapes)
                    collapseLastDims(d, 1);
                collapseLastDims(domain, 1);
            } else {
                break;
            }
        }
        return domain.get_shape();
    };
    const auto& tmpShape = master_shape.get_shape();
    fullWorkAmount = std::accumulate(tmpShape.begin(), tmpShape.end(), 1, std::multiplies<size_t>());
    exec_domain = findDimsToCollapse(master_shape, fullWorkAmount);
}
void Snippet::normalizeShapes() {
    auto edgeToBlockedShape = [](const EdgePtr& edge) {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        ngraph::Shape shape(blockedDesc->getBlockDims());
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
        return ngraph::snippets::op::Subgraph::BlockedShape{shape, blocking, precision};
    };
    auto prependWithOnes = [this](const PartialShape& dims) {
        if (tensorRank <= dims.size())
            return dims;
        std::vector<ov::Dimension> result(tensorRank, 1);
        std::copy(dims.begin(), dims.end(), &result[tensorRank - dims.size()]);
        return PartialShape {result};
    };
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
    for (size_t i = 0; i < inputShapes.size(); i++)
        input_blocked_shapes.push_back(edgeToBlockedShape(getParentEdgesAtPort(i)[0]));

    ngraph::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
    for (size_t i = 0; i < outputShapes.size(); i++)
        output_blocked_shapes.push_back(edgeToBlockedShape(getChildEdgesAtPort(i)[0]));
    master_shape = snippet->canonicalize(output_blocked_shapes, input_blocked_shapes);
    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), master_shape.size());
    // Canonicalization broadcasts inputs and outputs to max input rank, which can be smaller than tensorRank
    // prepend to enable 6D scheduler
    master_shape = prependWithOnes(master_shape);
    const auto &body = snippet->get_body();
    for (const auto& p : body->get_parameters()) {
        bodyInputShapes.emplace_back(prependWithOnes(p->get_output_partial_shape(0)));
    }
}
void Snippet::createPrimitive() {
    // determine canonicalize, determine master_shape and prepend up to 6D
    // NB! bodyInputShapes are updated, so body reshape might be needed
    normalizeShapes();
    if (master_shape.is_static()) {
        for (int i = 0; i < tileRank; i++) {
            schedulerWorkAmount = fullWorkAmount / exec_domain[exec_domain.size() - 1 - i];
            exec_domain[exec_domain.size() - 1 - i] = 1;
        }
        prepareParams();
        jit_snippets_compile_args jcp;
        jcp.tileRank = tileRank;
        std::copy(data_offsets.begin(), data_offsets.end(), jcp.data_offsets);
        std::copy(scheduler_offsets.begin(), scheduler_offsets.end(), jcp.scheduler_offsets);
        // code generation part
        // it might be worth to generate explicitly for scheduler work amount for now,
        // but in future some interface should be defined in order to communicate schedule for a kernel
        // or generate schedule for a kernel.
        // Here kernel is generated for most warying dimension by default.
        generate(&jcp);
    } else {
        generate(nullptr);
    }
}

void Snippet::prepareParams() {
    // here must be all the stuff that could only be done for static shapes, e.g. offset calculation
    // Here it must be all the stuff that could be done once for both static and dynamic shapes
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    dataSize = config.inConfs[0].getMemDesc()->getPrecision().size();

    optimizeExecDomain();
    // todo: do we need this reshape or it's better to pass scheduler_dims as compile-time args?
    std::map<size_t , ov::PartialShape> updated_shapes;
    for (size_t i = 0; i < bodyInputShapes.size(); i++)
        updated_shapes[i] = ov::PartialShape(bodyInputShapes[i]);
    snippet->get_body()->reshape(updated_shapes);
    master_shape = snippet->get_master_shape();

    calcJITParams(data_offsets, scheduler_offsets);
    auto initStartMemoryOffsets = [this]() {
        const auto config = getSelectedPrimitiveDescriptor()->getConfig();
        const size_t numInputs = inputShapes.size();
        start_offset_in.resize(numInputs);
        srcMemPtrs.resize(numInputs);
        for (size_t i = 0; i < numInputs; i++) {
            const auto memPtr = getParentEdgeAt(i)->getMemoryPtr();
            srcMemPtrs[i] = memPtr;
            start_offset_in[i] =  memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }
        const size_t numOutputs = outputShapes.size();
        start_offset_out.resize(numOutputs);
        dstMemPtrs.resize(numOutputs);
        for (size_t i = 0; i < numOutputs; i++) {
            const auto memPtr = getChildEdgeAt(i)->getMemoryPtr();
            dstMemPtrs[i] = memPtr;
            start_offset_out[i] = memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }
    };
    // initialize start offsets to src and dst memory
    // Needs to be done for every set of input shapes sce memory ptrs could've updated
    initStartMemoryOffsets();
    for (int i = 0; i < tileRank; i++) {
        schedulerWorkAmount = fullWorkAmount / exec_domain[exec_domain.size() - 1 - i];
        exec_domain[exec_domain.size() - 1 - i] = 1;
    }
}

bool Snippet::needPrepareParams() const {
    return (schedule.ptr == nullptr);
}

void Snippet::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr || !canUseOptimizedImpl) {
        IE_THROW() << "Snippet can't use Optimized implementation and can't fallback to reference";
    }
    jit_snippets_call_args call_args;
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->GetData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->GetData()) + start_offset_out[i];

    if (tensorRank == rank6D) {
        schedule_6d(call_args);
    } else {
        schedule_nt(call_args);
    }
}

void Snippet::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Snippet::created() const {
    return getType() == Type::Subgraph;
}

bool Snippet::canBeInPlace() const {
    if (getParentEdgesAtPort(0)[0]->getParent()->getType() == Type::Input) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

//void Snippet::define_schedule() {
//    auto edgeToBlockedShape = [](const EdgePtr& edge) {
//        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
//        ngraph::Shape shape(blockedDesc->getBlockDims());
//        ngraph::AxisVector blocking(blockedDesc->getOrder());
//        ngraph::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
//        return ngraph::snippets::op::Subgraph::BlockedShape{shape, blocking, precision};
//    };
//    auto prependWithOnes = [this](const PartialShape& dims) {
//        if (tensorRank <= dims.size())
//            return dims;
//        std::vector<ov::Dimension> result(tensorRank, 1);
//        std::copy(dims.begin(), dims.end(), &result[tensorRank - dims.size()]);
//        return PartialShape {result};
//    };
//    ngraph::snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
//    for (size_t i = 0; i < inputShapes.size(); i++)
//        input_blocked_shapes.push_back(edgeToBlockedShape(getParentEdgesAtPort(i)[0]));
//
//    ngraph::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
//    for (size_t i = 0; i < outputShapes.size(); i++)
//        output_blocked_shapes.push_back(edgeToBlockedShape(getChildEdgesAtPort(i)[0]));
//    master_shape = snippet->canonicalize(output_blocked_shapes, input_blocked_shapes);
//    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
//    tensorRank = std::max(static_cast<size_t>(rank6D), master_shape.size());
//    // Canonicalization broadcasts inputs and outputs to max input rank, which can be smaller than tensorRank
//    // prepend to enable 6D scheduler
//    master_shape = prependWithOnes(master_shape);
//    const auto &body = snippet->get_body();
//    for (const auto& p : body->get_parameters()) {
//        bodyInputShapes.emplace_back(prependWithOnes(p->get_shape()));
//    }
//
//    auto findDimsToCollapse = [this]() -> int {
//        auto collapseLastDims = [](PartialShape& dims, size_t dimsToCollapse) {
//            if (dimsToCollapse >= dims.size() - 1)
//                IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
//            for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
//                dims[dims.size() - 1] *= dims[i];
//            }
//
//            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
//                dims[i] = dims[i - dimsToCollapse];
//            }
//
//            for (int i = dimsToCollapse - 1; i >= 0; i--) {
//                dims[i] = 1;
//            }
//        };
//        int collapsedDims = 0;
//        size_t minimalConcurrency = parallel_get_max_threads();
//        size_t minimalJitWorkAmount = 256;
//        // If one of the last two dims is dynamic then skip collapsing and pack it to dynamic Tile
//        if (!master_shape[tensorRank - 1].is_static() || !master_shape[tensorRank - 2].is_static()) {
//            tileRank++;
//            return 0;
//        }
//        size_t currentJitWorkAmount = master_shape[tensorRank - 1].get_length();
//        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
//            if (static_cast<int>(master_shape.size()) - collapsedDims - 2 < 0)
//                break;
//
//            bool canCollapse = true;
//            for (size_t i = 0; i < bodyInputShapes.size(); i++) {
//                const size_t last = bodyInputShapes[i].size() - 1;
//                if ((bodyInputShapes[i][last - 1] != 1 && bodyInputShapes[i][last] == 1) ||
//                    (bodyInputShapes[i][last - 1] == 1 && bodyInputShapes[i][last] != 1)) {
//                    canCollapse = false;
//                    break;
//                }
//            }
//
//            size_t nextJitWorkAmount = currentJitWorkAmount * master_shape[master_shape.size() - 2].get_length();
//            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
//                currentJitWorkAmount = nextJitWorkAmount;
//                // if we cannot use dim collapsing we should use tile2D
//                if (!canCollapse) {
//                    if (tileRank < maxTileRank) {
//                        tileRank++;
//                        continue;
//                    }
//
//                    break;
//                }
//                collapsedDims++;
//                for (auto &d : bodyInputShapes)
//                    collapseLastDims(d, 1);
//                collapseLastDims(master_shape, 1);
//            } else {
//                break;
//            }
//        }
//        return collapsedDims;
//    };
//    batchDimIdx = tensorRank - exec_domain.size();
//    findDimsToCollapse();
//
//    std::map<size_t , ov::PartialShape> updated_shapes;
//    for (size_t i = 0; i < bodyInputShapes.size(); i++)
//        updated_shapes[i] = ov::PartialShape(bodyInputShapes[i]);
//    snippet->get_body()->reshape(updated_shapes);
//    master_shape = snippet->get_master_shape();
//    for (const auto &r : snippet->get_body()->get_results())
//        bodyOutputShapes.emplace_back(r->get_input_shape(0));
//    // we need fullWorkAmount only in the static case
//    if (master_shape.is_static()) {
//        exec_domain = master_shape.get_shape();
//        fullWorkAmount = std::accumulate(exec_domain.begin(), exec_domain.end(), 1, std::multiplies<size_t>());
//        for (int i = 0; i < tileRank; i++) {
//            schedulerWorkAmount = fullWorkAmount / exec_domain[exec_domain.size() - 1 - i];
//            exec_domain[exec_domain.size() - 1 - i] = 1;
//        }
//    }
//}

void Snippet::generate(const jit_snippets_compile_args* jcp) {
    size_t harness_num_dims = exec_domain.size() - tileRank;
    if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS) {
        canUseOptimizedImpl = false;
        harness_num_dims = SNIPPETS_MAX_HARNESS_DIMS;
    }
    schedule = snippet->generate(reinterpret_cast<const void*>(jcp));
}

void Snippet::schedule_6d(const jit_snippets_call_args& call_args) const {
    const auto& dom = exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            std::cerr << dom[0] << " " << dom[1] << " " << dom[2] << " " << dom[3] << " " << dom[4] << "\n";
            schedule.get_callable<kernel>()(indexes, &call_args);
        });
}

void Snippet::schedule_nt(const jit_snippets_call_args& call_args) const {
    const auto& work_size = exec_domain;
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(schedulerWorkAmount, nthr, ithr, start, end);

        std::vector<int64_t> indexes(work_size.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = work_size.size() - 2; j >= 0; j--) {
                indexes[j] = tmp % work_size[j];
                tmp /= work_size[j];
            }

            schedule.get_callable<kernel>()(indexes.data(), &call_args);
        }
    });
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
