/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/TradeEvent.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::event::TradeEvent>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::event::TradeEvent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "trade") {
                auto trade = std::make_shared<Trade>();
                val.convert(*trade);
                v.trade = trade;
            }
            else if (key == "ctx") {
                v.ctx = val.as<TradeContext>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::event::TradeEvent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::TradeEvent& v) const
    {
        if constexpr (std::same_as<Stream, taosim::serialization::HumanReadableStream>) {
            o.pack_map(12);

            o.pack("y");
            o.pack("t");

            o.pack("i");
            o.pack(v.trade->m_id);

            o.pack("s");
            o.pack(std::to_underlying(v.trade->m_direction));

            o.pack("t");
            o.pack(v.trade->m_timestamp);
            
            o.pack("q");
            o.pack(v.trade->m_volume);

            o.pack("p");
            o.pack(v.trade->m_price);
        
            o.pack("Ti");
            o.pack(v.trade->m_aggressingOrderID);

            o.pack("Ta");
            o.pack(v.ctx.aggressingAgentId);

            o.pack("Tf");
            o.pack(v.ctx.fees.taker);

            o.pack("Mi");
            o.pack(v.trade->m_restingOrderID);

            o.pack("Ma");
            o.pack(v.ctx.restingAgentId);

            o.pack("Mf");
            o.pack(v.ctx.fees.maker);
        }
        else if constexpr (std::same_as<Stream, taosim::serialization::BinaryStream>) {
            o.pack_map(3);

            o.pack("event");
            o.pack("trade");

            o.pack("trade");
            o.pack(v.trade);

            o.pack("ctx");
            o.pack(v.ctx);
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

