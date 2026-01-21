/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/util/SubscriptionRegistry.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<typename T>
struct convert<taosim::util::SubscriptionRegistry<T>>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::util::SubscriptionRegistry<T>& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& val : o.via.array) {
            v.add(val.as<T>());
        }

        return o;
    }
};

template<typename T>
struct pack<taosim::util::SubscriptionRegistry<T>>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::util::SubscriptionRegistry<T>& v) const
    {
        o.pack_array(v.subs().size());

        for (auto&& sub : v.subs()) {
            o.pack(sub);
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------