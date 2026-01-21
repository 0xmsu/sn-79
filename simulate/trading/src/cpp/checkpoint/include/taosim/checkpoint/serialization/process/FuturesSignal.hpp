/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/FuturesSignal.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::process::FuturesSignal>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::process::FuturesSignal& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "lastCount") {
                val.convert(v.state().lastCount);
            }
            else if (key == "lastSeed") {
                val.convert(v.state().lastSeed);
            }
            else if (key == "lastSeedTime") {
                val.convert(v.state().lastSeedTime);
            }
            else if (key == "value") {
                val.convert(v.state().value);
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::process::FuturesSignal>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::FuturesSignal& v) const
    {
        o.pack_map(4);

        o.pack("lastCount");
        o.pack(v.state().lastCount);

        o.pack("lastSeed");
        o.pack(v.state().lastSeed);

        o.pack("lastSeedTime");
        o.pack(v.state().lastSeedTime);

        o.pack("value");
        o.pack(v.state().value);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------