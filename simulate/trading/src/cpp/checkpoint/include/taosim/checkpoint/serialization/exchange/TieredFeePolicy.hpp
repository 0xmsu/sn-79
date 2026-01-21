/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/TieredFeePolicy.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::exchange::TieredFeePolicy>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::TieredFeePolicy& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "agentTiers") {
                using T = std::remove_cvref_t<decltype(v.agentTiers())>;
                v.agentTiers() = val.as<T>();
            }
            else if (key == "agentVolumes") {
                using T = std::remove_cvref_t<decltype(v.agentVolumes())>;
                v.agentVolumes() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::exchange::TieredFeePolicy>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::TieredFeePolicy& v) const
    {
        o.pack_map(3);

        o.pack("type");
        o.pack("tiered");

        o.pack("agentTiers");
        o.pack(v.agentTiers());

        o.pack("agentVolumes");
        o.pack(v.agentVolumes());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
