/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/L3RecordContainer.hpp>
#include <taosim/event/serialization/CancellationEvent.hpp>
#include <taosim/event/serialization/OrderEvent.hpp>
#include <taosim/event/serialization/TradeEvent.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::event::L3Record::Entry>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::event::L3Record::Entry& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto eventType = taosim::serialization::msgpackFindMap<std::string_view>(o, "event");
        if (!eventType) {
            throw taosim::serialization::MsgPackError{};
        }

        if (eventType == "cancel") {
            v = o.as<taosim::event::CancellationEvent>();
        }
        else if (eventType == "place") {
            v = o.as<taosim::event::OrderEvent>();
        }
        else if (eventType == "trade") {
            v = o.as<taosim::event::TradeEvent>();
        }
        else {
            throw taosim::serialization::MsgPackError{};
        }

        return o;
    }
};

template<>
struct pack<taosim::event::L3Record::Entry>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::L3Record::Entry& v) const
    {
        std::visit([&](auto&& entry) { o.pack(entry); }, v);
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------

