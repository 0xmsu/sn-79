/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/FuturesTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/common.hpp>
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
struct convert<taosim::agent::FuturesTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::FuturesTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "volumeFactor") {
                using T = std::remove_cvref_t<decltype(v.volumeFactor())>;
                v.volumeFactor() = val.as<T>();
            }
            else if (key == "factorCounter") {
                using T = std::remove_cvref_t<decltype(v.factorCounter())>;
                v.factorCounter() = val.as<T>();
            }
            else if (key == "lastUpdate") {
                using T = std::remove_cvref_t<decltype(v.lastUpdate())>;
                v.lastUpdate() = val.as<T>();
            }
            else if (key == "orderFlag") {
                using T = std::remove_cvref_t<decltype(v.orderFlag())>;
                v.orderFlag() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::FuturesTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::FuturesTraderAgent& v) const
    {
        o.pack_map(4);

        o.pack("volumeFactor");
        o.pack(v.volumeFactor());

        o.pack("factorCounter");
        o.pack(v.factorCounter());

        o.pack("lastUpdate");
        o.pack(v.lastUpdate());

        o.pack("orderFlag");
        o.pack(v.orderFlag());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------