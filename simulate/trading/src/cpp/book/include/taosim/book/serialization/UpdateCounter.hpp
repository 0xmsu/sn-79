/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/UpdateCounter.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::UpdateCounter>
{
    msgpack::object const& operator()(
        msgpack::object const& o, taosim::book::UpdateCounter& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "internalPeriod") {
                v.internalPeriod() = val.as<Timestamp>();
            }
            else if (key == "counter") {
                v.counter() = val.as<Timestamp>();
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::book::UpdateCounter>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::book::UpdateCounter& v) const
    {
        o.pack_map(2);

        o.pack("internalPeriod");
        o.pack(v.internalPeriod());

        o.pack("counter");
        o.pack(v.counter());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
