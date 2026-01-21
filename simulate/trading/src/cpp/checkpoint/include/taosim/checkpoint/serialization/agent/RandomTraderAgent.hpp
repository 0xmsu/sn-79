/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/RandomTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/common.hpp>
#include <taosim/serialization/msgpack/boost/circular_buffer.hpp>
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
struct convert<taosim::agent::RandomTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::RandomTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "topLevel") {
                using T = std::remove_cvref_t<decltype(v.topLevel())>;
                v.topLevel() = val.as<T>();
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
struct pack<taosim::agent::RandomTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::RandomTraderAgent& v) const
    {
        o.pack_map(2);

        o.pack("topLevel");
        o.pack(v.topLevel());

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