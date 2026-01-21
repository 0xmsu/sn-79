/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/RNG.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::process::RNG>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::process::RNG& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto seed = taosim::serialization::msgpackFindMap<uint64_t>(o, "seed");
        if (!seed) {
            throw taosim::serialization::MsgPackError{};
        }
        v = taosim::process::RNG{*seed};

        const auto callCount = taosim::serialization::msgpackFindMap<uint32_t>(o, "callCount");
        if (!callCount) {
            throw taosim::serialization::MsgPackError{};
        }
        v.discard(*callCount);
        
        return o;
    }
};

template<>
struct pack<taosim::process::RNG>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::RNG& v) const
    {
        o.pack_map(2);

        o.pack("callCount");
        o.pack(v.callCount());

        o.pack("seed");
        o.pack(v.seedValue());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------