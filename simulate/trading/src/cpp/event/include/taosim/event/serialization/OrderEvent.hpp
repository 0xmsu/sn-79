/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/OrderEvent.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::event::OrderEvent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::event::OrderEvent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "id") {
                v.id = val.as<OrderID>();
            }
            else if (key == "timestamp") {
                v.timestamp = val.as<Timestamp>();
            }
            else if (key == "volume") {
                v.volume = val.as<taosim::decimal_t>();
            }
            else if (key == "leverage") {
                v.leverage = val.as<taosim::decimal_t>();
            }
            else if (key == "direction") {
                v.direction = val.as<OrderDirection>();
            }
            else if (key == "stpFlag") {
                v.stpFlag = val.as<STPFlag>();
            }
            else if (key == "ctx") {
                v.ctx = val.as<OrderContext>();
            }
            else if (key == "postOnly") {
                v.postOnly = val.as<std::optional<bool>>();
            }
            else if (key == "timeInForce") {
                v.timeInForce = val.as<std::optional<taosim::TimeInForce>>();
            }
            else if (key == "expiryPeriod") {
                v.expiryPeriod = val.as<std::optional<std::optional<Timestamp>>>();
            }
            else if (key == "currency") {
                v.currency = val.as<Currency>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::event::OrderEvent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::OrderEvent& v) const
    {
        if constexpr (std::same_as<Stream, taosim::serialization::HumanReadableStream>) {
            o.pack_map(8);

            o.pack("y");
            o.pack("o");

            o.pack("i");
            o.pack(v.id);

            o.pack("c");
            o.pack(v.ctx.clientOrderId);

            o.pack("t");
            o.pack(v.timestamp);

            o.pack("q");
            o.pack(v.volume);

            o.pack("s");
            o.pack(std::to_underlying(v.direction));

            o.pack("p");
            o.pack(v.price);

            o.pack("l");
            o.pack(v.leverage);
        }
        else if constexpr (std::same_as<Stream, taosim::serialization::BinaryStream>) {
            o.pack_map(13);

            o.pack("event");
            o.pack("place");

            o.pack("id");
            o.pack(v.id);

            o.pack("timestamp");
            o.pack(v.timestamp);

            o.pack("volume");
            o.pack(v.volume);

            o.pack("leverage");
            o.pack(v.leverage);

            o.pack("direction");
            o.pack(v.direction);

            o.pack("stpFlag");
            o.pack(v.stpFlag);

            o.pack("price");
            o.pack(v.price);

            o.pack("ctx");
            o.pack(v.ctx);

            o.pack("postOnly");
            o.pack(v.postOnly);

            o.pack("timeInForce");
            o.pack(v.timeInForce);

            o.pack("expiryPeriod");
            o.pack(v.expiryPeriod);

            o.pack("currency");
            o.pack(v.currency);
        }
        else {
            static_assert(false, "Unrecognized Stream type");
        }
    
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------