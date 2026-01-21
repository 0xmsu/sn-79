/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/NoiseTraderAgent.hpp>
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
struct convert<taosim::agent::NoiseTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::NoiseTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        auto& s = v.state();

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "orderFlag") {
                using T = std::remove_cvref_t<decltype(s.orderFlag)>;
                s.orderFlag = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::NoiseTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::NoiseTraderAgent& v) const
    {
        const auto& s = v.state();

        o.pack_map(1);

        o.pack("orderFlag");
        o.pack(s.orderFlag);

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------