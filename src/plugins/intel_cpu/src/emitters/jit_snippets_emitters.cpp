// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {
jit_container_emitter::jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                      const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(const std::vector<size_t> &vec_pool,  const std::vector<size_t> &gpr_pool,
                                                    std::set<size_t>& vecs_used, std::set<size_t>& gprs_used) {
    if (body.empty())
        IE_THROW() << "Cannot map registers for jit_container_emitter when its body is empty";
    auto abstract_to_physical = [](const std::vector<size_t>& abstract_regs, const std::vector<size_t>& regs_pool) {
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++)
            physical_regs[i] = regs_pool.at(abstract_regs[i]);
        return physical_regs;
    };
    for (auto& code : body) {
        const auto& emitter = code.first;
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = code.second;
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                // Note that gpr_to_gpr is used for high-level utility operations like Kernel/TileScheduler/Tile.
                // Input registers are not mapped in this case, since they contain utility info
                // (num_params, tile increment, etc.), but not reg indexes.
                in_physical_regs = std::move(in_abstract_regs);
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, gpr_pool));
                gprs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, gpr_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec_pool));
                gprs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                vecs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, gpr_pool));
                vecs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                gprs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec_pool));
                vecs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                vecs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            default:
                IE_THROW() << "Unhandled in_out type";
        }
        code.second = std::make_pair(in_physical_regs, out_physical_regs);
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(code.first))
            container->map_abstract_registers(vec_pool, gpr_pool, vecs_used, gprs_used);
    }
}

KernelEmitter::KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
    if (!kernel)
        IE_THROW() << "KernelEmitter invoked with invalid op argument";
    if (kernel->region.empty())
        IE_THROW() << "KernelEmitter invoked with empty body";
    body = kernel->region;
    if (!kernel->compile_params)
        IE_THROW() << "KernelEmitter invoked without compile_params";
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    std::iota(gp_regs_pool.begin(), gp_regs_pool.end(), 0);
    std::iota(vec_regs_pool.begin(), vec_regs_pool.end(), 0);
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         static_cast<size_t>(dnnl::impl::cpu::x64::abi_param1.getIdx()),
                                         static_cast<size_t>(dnnl::impl::cpu::x64::abi_param2.getIdx())});
    std::set<size_t> vecs_used, gprs_used;
    map_abstract_registers(vec_regs_pool, gp_regs_pool, vecs_used, gprs_used);
    remove_regs_from_pool(gp_regs_pool, gprs_used);
    remove_regs_from_pool(vec_regs_pool, vecs_used);
    // Remember used gprs to pass it to the TileSchedulerEmitter, so it can init them with appropriate data ptrs
    gp_regs_used = std::vector<size_t>(gprs_used.begin(), gprs_used.end());
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out,
                              const std::vector<size_t> &pool,
                              const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool,
                                       const std::vector<size_t> &gpr) const {
    if (in.size() != 2)
        IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 2, got " << in.size();
    if (!out.empty())
        IE_THROW() << "KKernelEmitter got invalid number of outputs. Expected 0, got " << out.size();
}

void KernelEmitter::init_data_pointers(size_t num_inputs, size_t num_params,
                                              const Reg64& reg_indexes, const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    const int64_t harness_num_dims = jcp.master_shape.size() - 1;
    auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t *offsets, Reg64 reg_tmp) {
        for (int j = 0; j < harness_num_dims; j++) {
            if (jcp.master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    for (auto i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        // we can use the last data_ptr_reg as tmp_reg until the last iteration, and reg_const_params then
        Reg64 reg_tmp = i < num_params-1 ? data_ptr_regs.back() : reg_const_params;
        init_ptrs_with_offsets(data_ptr_regs[i], &jcp.data_offsets[i * harness_num_dims], reg_tmp);
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& allocated_vec_regs,
                              const std::vector<size_t>& allocated_gp_regs,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    h->preamble();

    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];

    Reg64 reg_indexes = Reg64(dnnl::impl::cpu::x64::abi_param1.getIdx());
    Reg64 reg_const_params = Reg64(dnnl::impl::cpu::x64::abi_param2.getIdx());
    std::vector<Reg64> data_ptr_regs(gp_regs_used.size());
    std::transform(gp_regs_used.begin(), gp_regs_used.end(), data_ptr_regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});

    init_data_pointers(num_inputs, num_inputs + num_outputs, reg_indexes, reg_const_params, data_ptr_regs);
    // todo: emit_impl is a const method, so we can't just push_back unused regs to the gp_regs_pool.
    //  we need a more elegant approach to avoid a full copy here
    auto local_gpr_pool = gp_regs_pool;
    local_gpr_pool.push_back(static_cast<size_t>(reg_indexes.getIdx()));
    local_gpr_pool.push_back(static_cast<size_t>(reg_const_params.getIdx()));
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        if (auto tile_scheduler = std::dynamic_pointer_cast<TileSchedulerEmitter>(emitter))
            out_regs = gp_regs_used;
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, local_gpr_pool);
    }
    h->postamble();
}

TileSchedulerEmitter::TileSchedulerEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto tile_scheduler = ov::as_type_ptr<ngraph::snippets::op::TileScheduler>(n);
    if (!tile_scheduler)
        IE_THROW() << "TileSchedulerEmitter invoked with invalid op argument";
    is_static = tile_scheduler->is_static;
    if (is_static && !tile_scheduler->compile_params)
        IE_THROW() << "TileEmitter invoked without compile_params";
    body = {tile_scheduler->vector_region, tile_scheduler->scalar_region};
    if (is_static)
        jcp = *reinterpret_cast<const jit_snippets_compile_args*>(tile_scheduler->compile_params);
}
void TileSchedulerEmitter::emit_code(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}
void TileSchedulerEmitter::validate_arguments(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    if (in.size() != 3)
        IE_THROW() << "TileSchedulerEmitter got invalid number of inputs. Expected 3, got " << in.size();
    if (out.size() != in[0] + in[1])
        IE_THROW() << "TileSchedulerEmitter got invalid number of outputs. Expected " << in[0] + in[1] << " , got " << out.size();
    if (body.size() != 2)
        IE_THROW() << "TileSchedulerEmitter got invalid body size, expected 2 (vector & scalar TileEmitter), got " << body.size();
    if (!(std::dynamic_pointer_cast<TileEmitter>(body[0].first) && std::dynamic_pointer_cast<TileEmitter>(body[1].first)))
        IE_THROW() << "TileSchedulerEmitter can contain only TileEmitters inside its body";
}

void TileSchedulerEmitter::emit_tiles(const Reg64& reg_inner_amount, const std::vector<Reg64>& data_ptr_regs, size_t vector_size,
                                      const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    // TileAllocatedEmitter is just an alias to perform dynamic_pointer_cast only once and reuse it below several times
    using TileAllocatedEmitter = std::pair<std::shared_ptr<TileEmitter>, const ngraph::snippets::RegInfo&>;
    TileAllocatedEmitter vector_tile {std::dynamic_pointer_cast<TileEmitter>(body[0].first), body[0].second};
    TileAllocatedEmitter scalar_tile {std::dynamic_pointer_cast<TileEmitter>(body[1].first), body[1].second};
    const size_t inner_work_amount = jcp.scheduler_work_amounts[1];
    auto process_tile =
        [&](const bool evaluate_once, const TileAllocatedEmitter& tile) {
            // If Tile is evaluated only once, then we can emit its body directly and skip work_amount decrements and checks
            if (evaluate_once) {
                tile.first->emit_body(vec_pool, gpr_pool);
            } else {
                std::vector<size_t> in_regs, out_regs;
                std::tie(in_regs, out_regs) = tile.second;
                // pass work_amount reg to Tile
                in_regs.push_back(static_cast<size_t>(reg_inner_amount.getIdx()));
                for (const auto& reg : data_ptr_regs)
                    out_regs.emplace_back(reg.getIdx());
                tile.first->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
            }
        };
    // todo: these optimizations should be performed on using Tile graph representation in the future
    bool vector_evaluate_once = false;
    if (inner_work_amount >= vector_size) {
        vector_evaluate_once = inner_work_amount < 2 * vector_size;
        // Need to set proper work amount for inner tiles if evaluated multiple times
        if (!vector_evaluate_once)
            h->mov(reg_inner_amount, inner_work_amount);
        process_tile(vector_evaluate_once, vector_tile);
    }
    if (inner_work_amount % vector_size >= 1) {
        bool scalar_evaluate_once = inner_work_amount % vector_size < 2;
        if (!scalar_evaluate_once) {
            // vector_tile is not executed, work_amount is not set
            if (inner_work_amount < vector_size) {
                h->mov(reg_inner_amount, inner_work_amount);
                // vector_tile is executed, but work_amount is neither set nor decremented appropriately.
            } else if (vector_evaluate_once) {
                vector_tile.first -> emit_ptr_increments(data_ptr_regs);
                h->mov(reg_inner_amount, inner_work_amount - vector_size);
            }
            // else: vector_tile is executed multiple times, so work_amount is already set
        }
        process_tile(scalar_evaluate_once, scalar_tile);
    }
}

void TileSchedulerEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& vec_pool,
                                     const std::vector<size_t>& gpr_pool,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    if (is_static)
        emit_static_impl(in, out, vec_pool, gpr_pool, emit_context);
    else
        emit_dynamic_impl(in, out, vec_pool, gpr_pool, emit_context);
}

void TileSchedulerEmitter::emit_static_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& vec_pool,
                                     const std::vector<size_t>& gpr_pool,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];
    const size_t vector_size = in[2];
    const size_t num_params = num_inputs + num_outputs;
    const auto& data_ptr_reg_idxs(out);
    std::vector<Reg64> data_ptr_regs(data_ptr_reg_idxs.size());
    std::transform(data_ptr_reg_idxs.begin(), data_ptr_reg_idxs.end(), data_ptr_regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});

    // todo: emit_impl has const input args, so we can't just pop_back necessary regs from gpr_pool.
    //  we need a more elegant approach to avoid a full copy here. Similar problem is demonstrated in KernelEmitter
    auto local_gpr_pool = gpr_pool;
    Reg64 reg_outer_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    Reg64 reg_inner_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    Label for_body;
    const size_t outer_work_amount = jcp.scheduler_work_amounts[0];
    if (outer_work_amount == 1) {
        // emit code directly without looping over external dim
        emit_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);
    } else if (outer_work_amount > 1) {
        // We need to create a Loop in this case
        h->mov(reg_outer_amount, outer_work_amount);
        h->L(for_body);
        {
            emit_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);

            // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
            //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
            //   To overcome this limitation, we add appropriate negative offsets if necessary.
            for (auto i = 0; i < num_params; i++) {
                if (jcp.scheduler_offsets[i] != 0) {
                    h->add(data_ptr_regs[i], jcp.scheduler_offsets[i]);
                }
            }
            // Note that outer dimensions are always incremented by 1 (outer tiles are always scalar)
            h->sub(reg_outer_amount, 1);
            h->cmp(reg_outer_amount, 1);
            h->jge(for_body, CodeGenerator::T_NEAR);
        }
    }
}

std::vector<AllocatedEmitter>& TileEmitter::get_nested_code() {
    return body;
}

void TileSchedulerEmitter::emit_dynamic_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            const std::vector<size_t>& vec_pool,
                                            const std::vector<size_t>& gpr_pool,
                                            const ov::intel_cpu::emitter_context *emit_context) const {
    std::cerr << "Dynamic implementation is being compiled\n";
    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];
    const size_t vector_size = in[2];
    const size_t num_params = num_inputs + num_outputs;
    // Todo: is it safe to rely on 6? At least add a check in the node
    const int64_t master_rank = 6; //jcp.master_shape.size();
//    const size_t outer_work_amount = jcp.tileRank == 1 ? 1 : jcp.master_shape[master_rank - 2]; // sometimes outer WA could be 1 even if exec domain isn't
//    const size_t inner_work_amount = jcp.master_shape[master_rank - 1];
    const int64_t offsetRank = master_rank - 1;

    const auto& vector_tile = body[0];
    const auto& scalar_tile = body[1];
//    const auto& vector_tile_body = std::dynamic_pointer_cast<TileEmitter>(vector_tile.first)->get_nested_code();
//    const auto& scalar_tile_body = std::dynamic_pointer_cast<TileEmitter>(scalar_tile.first)->get_nested_code();

    // remove data ptr regs from the pool, since they should be preserved
    const auto& data_ptr_reg_idxs(out);

    // It is critical that reg_indexes and reg_const_params represent the
    // first two runtime arguments, since they are used to calculate offsets
    Reg64 reg_indexes = Reg64(static_cast<int>(in[3]));
    Reg64 reg_const_params = Reg64(static_cast<int>(in[4]));

    auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t offset_displ, Reg64 reg_tmp) {
        // Note that we don't need the last offset, since increment over the last dimension is
        // performed directly by the Load emitter
//        for (int k = 0; k < offsetRank; k++) {
//            if (jcp.master_shape[k] != 1 && offsets[k] != 0) {
//                h->mov(reg_tmp, offsets[k]);
//                h->imul(reg_tmp, h->ptr[reg_indexes + k * sizeof(size_t)]);
//                h->add(pointer, reg_tmp);
//            }
//        }
        const size_t data_offests_displ = GET_OFF(data_offsets) + offset_displ * sizeof(int64_t);
        // todo: we can pre-filter data_offsets so that only (master_shape[k] != 1 && offsets[k] != 0) are stored there
        //  but we'll need an additional index array to specify appropriate "k" values for every input
        //  * size_t num_non_zero_offsets[num_params] - specifies number of non-zero offsets for every input
        //  * size_t offsetted_indexes* - points to memory chunk sizeof(sum(num_non_zero_offsets) * sizeof(size_t)) -
        //                                  specifies indexes of input indexes (reg_index) that need an offset
        //  * size_t data_offsets* - the same size as offsetted_indexes - offset values for input indexes
        for (int k = 0; k < offsetRank; k++) {
                h->mov(reg_tmp, h->ptr[reg_const_params + data_offests_displ + k * sizeof(int64_t)]);
                h->imul(reg_tmp, h->ptr[reg_indexes + k * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
        }
    };
    std::vector<Reg64> data_ptr_regs(num_params);
    if (gpr_pool.size() == 0)
        IE_THROW() << "Dynamic TileSchedulerEmitter needs one more additional gpr";
    Reg64 reg_tmp = Reg64(static_cast<int>(gpr_pool.back()));
    for (int j = 0; j < num_params; j++) {
        data_ptr_regs[j] = Reg64(static_cast<int>(data_ptr_reg_idxs[j]));
        // todo: Maybe it's more convenient to pass src & dst pointers in a single array
        auto ptr_offset = j < num_inputs ? GET_OFF(src_ptrs) : GET_OFF(dst_ptrs) - num_inputs * sizeof(void*);
        h->mov(data_ptr_regs[j], h->ptr[reg_const_params + ptr_offset + j * sizeof(void*)]);
        // todo: you can't use this trick for dynamic tile
        // we can use the last data_ptr_reg as tmp_reg until the last iteration, and reg_const_params then
//        Reg64 reg_tmp = j < num_params - 1 ? Reg64(static_cast<int>(data_ptr_reg_idxs.back())) : reg_const_params;
//        init_ptrs_with_offsets(data_ptr_regs[j], jcp.data_offsets + j * offsetRank, reg_tmp);
        init_ptrs_with_offsets(data_ptr_regs[j], j * offsetRank, reg_tmp);
    }
    Label for_body, single_outer_tile, end;
    // We don't need runtime args, since the data_ptrs are already initialized,
    // So let's reuse reg_indexes and reg_const_params to store work amounts
    Reg64 reg_outer_amount = reg_indexes;
    Reg64 reg_inner_amount = reg_const_params;
    auto emit_tiles = [&]() {
        // the minimal requirement is that tile (vector or scalar) is emitted only if it has some work to do (>= 1 iterations)
        auto process_tile =
            [&](const size_t tile_increment, const AllocatedEmitter& tile) {
                    Label tile_end;
                    h->cmp(reg_inner_amount, tile_increment);
                    h->jl(tile_end, CodeGenerator::T_NEAR);
                    std::vector<size_t> in_regs, out_regs;
                    std::tie(in_regs, out_regs) = tile.second;
                    // pass work_amount reg to Tile
                    in_regs.push_back(static_cast<size_t>(reg_inner_amount.getIdx()));
                    tile.first->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
                    h->L(tile_end);
            };
        h->mov(reg_inner_amount, h->ptr[reg_const_params + GET_OFF(scheduler_work_amounts) + sizeof(size_t)]);
        process_tile(vector_size, vector_tile);
        process_tile(1, scalar_tile);
    };

    {
        h->mov(reg_outer_amount, h->ptr[reg_const_params + GET_OFF(scheduler_work_amounts)]);
        // We don't need to apply scheduler offsets, or update reg_outer_amount in case of outer WA == 1
        h->cmp(reg_outer_amount, 1);
        h->je(single_outer_tile, CodeGenerator::T_NEAR);
        //
        h->L(for_body);
        {
            emit_tiles();

            // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
            //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
            //   To overcome this limitation, we add appropriate negative offsets if necessary.
            for (auto i = 0; i < num_params; i++) {
                // NB! many scheduler offsets are zero
                h->add(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(scheduler_offsets) + i * sizeof(int64_t)]);
            }
            // Note that outer dimensions are always incremented by 1 (outer tiles are always scalar)
            h->sub(reg_outer_amount, 1);
            h->cmp(reg_outer_amount, 1);
            h->jge(for_body, CodeGenerator::T_NEAR);
            h->jmp(end, CodeGenerator::T_NEAR);
        }
        h->L(single_outer_tile);
        {
            // emit code directly without looping over external dim and applying scheduler offsets
            emit_tiles();
        }
        h->L(end);
    }
}

TileEmitter::TileEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto tile = ov::as_type_ptr<ngraph::snippets::op::Tile>(n);
    if (!tile)
        IE_THROW() << "TileEmitter invoked with invalid op argument";
    body = tile->region;
    if (body.empty())
        IE_THROW() << "TileEmitter is invoked with empty body";
    num_inputs = tile->num_inputs;
    num_outputs = tile->num_outputs;
    io_dims = tile->io_dims;
    increment = tile->increment;
    if (io_dims.size() != num_inputs + num_outputs)
        IE_THROW() << "TileEmitter constructor got inconsistent arguments. Check num_inputs + num_outputs == io_dims.size()";
}

void TileEmitter::emit_code(const std::vector<size_t> &in,
                            const std::vector<size_t> &out,
                            const std::vector<size_t> &pool,
                            const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileEmitter::validate_arguments(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    if (in.size() != 1)
        IE_THROW() << "TileEmitter got invalid number of inputs. Expected 1, got " << in.size();
    if (out.size() != io_dims.size())
        IE_THROW() << "TileEmitter got invalid number of outputs. Expected " << io_dims.size() << " , got " << out.size();
}

void TileEmitter::emit_body(const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    for (auto& code : body)
        code.first->emit_code(code.second.first, code.second.second, vec_pool, gpr_pool);
}

void TileEmitter::emit_ptr_increments(const std::vector<Reg64>& data_ptr_regs) const {
    for (size_t i = 0; i < num_inputs; i++) {
        // those with dims == 1 will be broadcasted, hence don't require increment
        if (io_dims[i] != 1)
            h->add(data_ptr_regs[i], increment * sizeof(float));
    }
    for (size_t i = num_inputs; i < num_inputs + num_outputs; i++)
        h->add(data_ptr_regs[i], increment * sizeof(float));
}

void TileEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& vec_pool,
                            const std::vector<size_t>& gpr_pool,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    Reg64 work_amount = Reg64(static_cast<int>(in[0]));
    std::vector<Reg64> data_ptr_regs(out.size());
    std::transform(out.begin(), out.end(), data_ptr_regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
    Label for_body;
    // Note that:
    // * Work amount must be set by TileScheduler that executes Tiles
    // * TileScheduler executes Tile only if it has to perform >= 1 iterations
    h->L(for_body);
    emit_body(vec_pool, gpr_pool);
    emit_ptr_increments(data_ptr_regs);
    h->sub(work_amount, increment);
    h->cmp(work_amount, increment);
    h->jge(for_body, CodeGenerator::T_NEAR);
}

FakeBroadcastEmitter::FakeBroadcastEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    if (n->get_input_shape(0).empty())
        use_broadcast = true;
    else if (*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin())
        use_broadcast = true;
    else
        use_broadcast = false;
}

void FakeBroadcastEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out,
          const std::vector<size_t>& pool,
          const std::vector<size_t>& gpr,
          const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void FakeBroadcastEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    if (use_broadcast) {
        h->uni_vbroadcastss(vmm_dst, Xmm(in[0]));
    } else {
        h->uni_vmovups(vmm_dst, vmm_src0);
    }
}

ScalarEmitter::ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    value = dnnl::impl::cpu::x64::float2int(ov::as_type_ptr<ngraph::snippets::op::Scalar>(n)->cast_vector<float>()[0]);
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& pool,
                              const std::vector<size_t>& gpr,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_dst  = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}


MemoryEmitter::MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out,
                             const std::vector<size_t>& pool,
                             const std::vector<size_t>& gpr,
                             const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 out_reg(static_cast<int>(out[0]));
    Vmm vmm_src0 = Vmm(in[0]);
    h->uni_vmovups(h->ptr[out_reg], vmm_src0);
}

ScalarStoreEmitter::ScalarStoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                       const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
}

void ScalarStoreEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out,
                                   const std::vector<size_t>& pool,
                                   const std::vector<size_t>& gpr,
                                   const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ScalarStoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 out_reg(static_cast<int>(out[0]));
    Xmm vmm_src0 = Xmm(in[0]);
    h->uni_vmovss(h->ptr[out_reg], vmm_src0);
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n)
                         : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(static_cast<int>(in[0]));
    Vmm vmm_src0 = Vmm(out[0]);
    h->uni_vmovups(vmm_src0, h->ptr[in_reg]);
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void BroadcastLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& pool,
                                     const std::vector<size_t>& gpr,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_src0 = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
}


ScalarLoadEmitter::ScalarLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                     const std::shared_ptr<ov::Node>& n)
                                    : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void ScalarLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                  const std::vector<size_t>& out,
                                  const std::vector<size_t>& pool,
                                  const std::vector<size_t>& gpr,
                                  const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ScalarLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(static_cast<int>(in[0]));
    Xmm vmm_src0 = Xmm(out[0]);
    h->uni_vmovss(vmm_src0, h->ptr[in_reg]);
}
}   // namespace intel_cpu
}   // namespace ov
