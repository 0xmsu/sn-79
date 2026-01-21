/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>

#include <range/v3/view/drop.hpp>

#include <boost/circular_buffer.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<typename T, typename Alloc>
struct convert<boost::circular_buffer<T, Alloc>>
{
    const msgpack::object& operator()(
        const msgpack::object& o, boost::circular_buffer<T, Alloc>& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        v.set_capacity(arr.ptr[0].as<size_t>());
        v.clear();

        for (const auto& val : arr | ranges::views::drop(1)) {
            v.push_back(val.as<T>());
        }

        return o;
    }
};

template<typename T, typename Alloc>
struct pack<boost::circular_buffer<T, Alloc>>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const boost::circular_buffer<T, Alloc>& v) const
    {
        o.pack_array(1 + v.size());

        o.pack(v.capacity());

        for (const auto& item : v) {
            o.pack(item);
        }

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------