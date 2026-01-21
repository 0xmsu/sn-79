/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>

#include <boost/bimap.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<typename L, typename R>
struct convert<boost::bimap<L, R>>
{
    const msgpack::object& operator()(const msgpack::object& o, boost::bimap<L, R>& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }
    
        for (const auto& [key, val] : o.via.map) {
            v.insert({key.as<L>(), val.as<R>()});
        }

        return o;
    }
};

template<typename L, typename R>
struct pack<boost::bimap<L, R>>
{
    template<typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, const boost::bimap<L, R>& v) const
    {
        o.pack_map(v.size());

        for (const auto& [left, right] : v) {
            o.pack(left);
            o.pack(right);
        }

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------