/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Account.hpp>
#include <taosim/checkpoint/serialization/accounting/Balances.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::Account>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::Account& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "holdings") {
                v.holdings() = val.as<taosim::accounting::Account::Holdings>();
            }
        }

        v.activeOrders().resize(v.holdings().size());

        return o;
    }
};

template<>
struct pack<taosim::accounting::Account>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::Account& v) const
    {
        o.pack_map(1);

        o.pack("holdings");
        o.pack(v.holdings());

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------