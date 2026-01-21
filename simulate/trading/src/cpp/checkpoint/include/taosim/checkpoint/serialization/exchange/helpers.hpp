/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/serialization/exchange/DynamicFeePolicy.hpp>
#include <taosim/checkpoint/serialization/exchange/TieredFeePolicy.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace taosim::checkpoint::serialization
{

//-------------------------------------------------------------------------

void packFeePolicy(auto& o, const taosim::exchange::FeePolicy& feePolicy)
{
    if (auto fp = dynamic_cast<const taosim::exchange::DynamicFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else if (auto fp = dynamic_cast<const taosim::exchange::TieredFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else {
        o.pack_nil();
    }
}

void unpackFeePolicy(const auto& o, taosim::exchange::FeePolicy& feePolicy)
{
    const auto typeOpt = taosim::serialization::msgpackFindMap<std::string_view>(o, "type");
    if (!typeOpt) {
        throw taosim::serialization::MsgPackError{};
    }
    auto type = *typeOpt;

    if (type == "dynamic") {
        auto ptr = dynamic_cast<taosim::exchange::DynamicFeePolicy*>(&feePolicy);
        o.convert(*ptr);
    }
    else if (type == "tiered") {
        auto ptr = dynamic_cast<taosim::exchange::TieredFeePolicy*>(&feePolicy);
        o.convert(*ptr);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::exchange::serialization

//-------------------------------------------------------------------------