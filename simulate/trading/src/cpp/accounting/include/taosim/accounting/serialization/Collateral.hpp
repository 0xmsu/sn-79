/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Collateral.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::Collateral>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::Collateral& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "base") {
                v.base() = val.as<taosim::decimal_t>();
            }
            else if (key == "quote") {
                v.quote() = val.as<taosim::decimal_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::accounting::Collateral>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::Collateral& v) const
    {
        o.pack_map(2);

        o.pack("base");
        o.pack(v.base());

        o.pack("quote");
        o.pack(v.quote());

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------