/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/MessagePayload.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<EmptyPayload>
{
    const msgpack::object& operator()(const msgpack::object& o, EmptyPayload& v)
    {
        if (!o.is_nil()) {
            throw taosim::serialization::MsgPackError{};
        }
        return o;
    }
};

template<>
struct pack<EmptyPayload>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const EmptyPayload& v) const
    {
        o.pack_nil();
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------