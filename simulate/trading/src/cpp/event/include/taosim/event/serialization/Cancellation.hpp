/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/Cancellation.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::event::Cancellation>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::event::Cancellation& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "orderId") {
                v.id = val.as<OrderID>();
            }
            else if (key == "volume") {
                v.volume = val.as<std::optional<taosim::decimal_t>>();
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::event::Cancellation>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::Cancellation& v) const
    {
        using namespace std::string_literals;

        o.pack_map(3);

        o.pack("event");
        o.pack("cancel");

        o.pack("orderId");
        o.pack(v.id);

        o.pack("volume");
        o.pack(v.volume);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------