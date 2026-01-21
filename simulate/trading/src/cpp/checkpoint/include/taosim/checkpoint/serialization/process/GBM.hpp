/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/GBM.hpp>
#include <taosim/checkpoint/serialization/process/RNG.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::process::GBM>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::process::GBM& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "rng") {
                v.state().rng = val.as<taosim::process::RNG>();
            }
            else if (key == "t") {
                v.state().t = val.as<double>();
            }
            else if (key == "W") {
                v.state().W = val.as<double>();
            }
            else if (key == "value") {
                v.val() = val.as<double>();
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::process::GBM>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::GBM& v) const
    {
        o.pack_map(4);

        o.pack("rng");
        o.pack(v.state().rng);

        o.pack("t");
        o.pack(v.state().t);

        o.pack("W");
        o.pack(v.state().W);

        o.pack("value");
        o.pack(v.val());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------