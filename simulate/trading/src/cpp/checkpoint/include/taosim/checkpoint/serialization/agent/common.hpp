/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
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
struct convert<taosim::agent::DelayBounds>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::DelayBounds& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.min);
        arr.ptr[1].convert(v.max);

        return o;
    }
};

template<>
struct pack<taosim::agent::DelayBounds>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::DelayBounds& v) const
    {
        o.pack_array(2);

        o.pack(v.min);
        o.pack(v.max);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::TimestampedPrice>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::TimestampedPrice& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.timestamp);
        arr.ptr[1].convert(v.price);

        return o;
    }
};

template<>
struct pack<taosim::agent::TimestampedPrice>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::TimestampedPrice& v) const
    {
        o.pack_array(2);

        o.pack(v.timestamp);
        o.pack(v.price);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::TopLevel>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::TopLevel& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.bid);
        arr.ptr[1].convert(v.ask);

        return o;
    }
};

template<>
struct pack<taosim::agent::TopLevel>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::TopLevel& v) const
    {
        o.pack_array(2);

        o.pack(v.bid);
        o.pack(v.ask);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::TopLevelWithVolumes>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::TopLevelWithVolumes& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.bid);
        arr.ptr[1].convert(v.ask);
        arr.ptr[2].convert(v.bidQty);
        arr.ptr[3].convert(v.askQty);

        return o;
    }
};

template<>
struct pack<taosim::agent::TopLevelWithVolumes>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::TopLevelWithVolumes& v) const
    {
        o.pack_array(4);

        o.pack(v.bid);
        o.pack(v.ask);
        o.pack(v.bidQty);
        o.pack(v.askQty);

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------