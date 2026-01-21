/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<DistributedAgentResponsePayload>
{
    const msgpack::object& operator()(const msgpack::object& o, DistributedAgentResponsePayload& v)
    {
        if (!o.is_nil()) {
            throw taosim::serialization::MsgPackError{};
        }

        return o;
    }
};

template<>
struct pack<DistributedAgentResponsePayload>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const DistributedAgentResponsePayload& v) const
    {
        using namespace std::string_literals;

        o.pack_map(2);

        o.pack("agentId"s);
        o.pack(v.agentId);

        o.pack("payload"s);
        o.pack(v.payload);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------