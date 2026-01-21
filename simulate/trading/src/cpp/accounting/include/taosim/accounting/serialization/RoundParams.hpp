/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/common.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::RoundParams>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::RoundParams& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.baseDecimals);
        arr.ptr[1].convert(v.quoteDecimals);

        return o;
    }
};

template<>
struct pack<taosim::accounting::RoundParams>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::RoundParams& v) const
    {
        o.pack_array(2);

        o.pack(v.baseDecimals);
        o.pack(v.quoteDecimals);

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------