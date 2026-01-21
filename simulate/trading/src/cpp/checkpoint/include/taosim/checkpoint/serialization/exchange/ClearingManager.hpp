/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/serialization/exchange/FeePolicyWrapper.hpp>
#include <taosim/exchange/ClearingManager.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::exchange::ClearingManager::MarginCallContext>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::ClearingManager::MarginCallContext& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "orderId") {
                v.orderId = val.as<OrderID>();
            }
            else if (key == "agentId") {
                v.agentId = val.as<AgentId>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::exchange::ClearingManager::MarginCallContext>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::ClearingManager::MarginCallContext& v) const
    {
        o.pack_map(2);

        o.pack("orderId");
        o.pack(v.orderId);

        o.pack("agentId");
        o.pack(v.agentId);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::exchange::ClearingManager>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::ClearingManager& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "marginBuys") {
                v.marginBuys() = val.as<taosim::exchange::ClearingManager::MarginCallContainer>();
            }
            else if (key == "marginSells") {
                v.marginSells() = val.as<taosim::exchange::ClearingManager::MarginCallContainer>();
            }
            else if (key == "feePolicy") {
                val.convert(*v.feePolicy());
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::exchange::ClearingManager>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::ClearingManager& v) const
    {
        o.pack_map(3);

        o.pack("marginBuys");
        o.pack(v.marginBuys());

        o.pack("marginSells");
        o.pack(v.marginSells());

        o.pack("feePolicy");
        o.pack(*v.feePolicy());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
