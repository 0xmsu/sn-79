/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/ExchangeSignals.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::exchange::ExchangeSignals>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::ExchangeSignals& v) const
    {
        if (o.type != msgpack::type::POSITIVE_INTEGER) {
            throw taosim::serialization::MsgPackError{};
        }

        o.convert(v.eventCounter);

        return o;
    }
};

template<>
struct pack<taosim::exchange::ExchangeSignals>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::ExchangeSignals& v) const
    {
        o.pack(v.eventCounter);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
