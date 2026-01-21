/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/MessageQueue.hpp>
#include <taosim/message/serialization/Message.hpp>
#include <taosim/message/serialization/helpers.hpp>
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
struct convert<taosim::message::PrioritizedMessage>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::message::PrioritizedMessage& v)
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "msg") {
                v.msg = Message::create(val.as<Message>());
            }
            else if (key == "marginCallId") {
                v.marginCallId = val.as<uint64_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::message::PrioritizedMessage>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::message::PrioritizedMessage& v) const
    {
        o.pack_map(2);

        o.pack("msg");
        o.pack(v.msg);
    
        o.pack("marginCallId");
        o.pack(v.marginCallId);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::message::PrioritizedMessageWithId>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::message::PrioritizedMessageWithId& v)
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "pmsg") {
                v.pmsg = val.as<taosim::message::PrioritizedMessage>();
            }
            else if (key == "id") {
                v.id = val.as<uint64_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::message::PrioritizedMessageWithId>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::message::PrioritizedMessageWithId& v) const
    {
        o.pack_map(2);

        o.pack("pmsg");
        o.pack(v.pmsg);
    
        o.pack("id");
        o.pack(v.id);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::message::MessageQueue>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::message::MessageQueue& v)
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "queue") {
                using T = std::remove_cvref_t<decltype(v.queue())>;
                v.queue() = T{T::CompareType{}, val.as<T::container_type>()};
            }
            else if (key == "idCounter") {
                v.idCounter() = val.as<uint64_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::message::MessageQueue>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::message::MessageQueue& v) const
    {
        o.pack_map(2);

        o.pack("queue");
        o.pack(v.queue().underlying());
    
        o.pack("idCounter");
        o.pack(v.idCounter());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------