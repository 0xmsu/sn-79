/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Balance.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::Balance>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::Balance& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "initial") {
                v.getInitial() = val.as<taosim::decimal_t>();
            }
            else if (key == "free") {
                v.getFree() = val.as<taosim::decimal_t>();
            }
            else if (key == "reserved") {
                v.getReserved() = val.as<taosim::decimal_t>();
            }
            else if (key == "total") {
                v.getTotal() = val.as<taosim::decimal_t>();
            }
            else if (key == "reservations") {
                v.getReservations() = val.as<taosim::accounting::Balance::Reservations>();
            }
            else if (key == "symbol") {
                v.getSymbol() = val.as<std::string>();
            }
            else if (key == "roundingDecimals") {
                v.getRoundingDecimals() = val.as<uint32_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::accounting::Balance>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::Balance& v) const
    {
        o.pack_map(7);

        o.pack("initial");
        o.pack(v.getInitial());

        o.pack("free");
        o.pack(v.getFree());

        o.pack("reserved");
        o.pack(v.getReserved());

        o.pack("total");
        o.pack(v.getTotal());
    
        o.pack("reservations");
        o.pack(v.getReservations());

        o.pack("symbol");
        o.pack(v.getSymbol());
    
        o.pack("roundingDecimals");
        o.pack(v.getRoundingDecimals());
    
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
