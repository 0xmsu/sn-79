/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/MagneticField.hpp>
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
struct convert<taosim::process::DurationComp>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::process::DurationComp& v) const
    {    
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }
        
        const auto& arr = o.via.array;

        arr.ptr[0].convert(v.delay);
        arr.ptr[1].convert(v.psi);

        return o;
    }
};

template<>
struct pack<taosim::process::DurationComp>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::DurationComp& v) const
    {
        o.pack_array(2);

        o.pack(v.delay);
        o.pack(v.psi);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::process::MagneticField>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::process::MagneticField& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }
        
        auto& s = v.state();

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "rng") {
                val.convert(s.rng);
            }
            else if (key == "logReturn") {
                val.convert(s.logReturn);
            }
            else if (key == "magnetism") {
                val.convert(s.magnetism);
            }
            else if (key == "magnetismReturn") {
                val.convert(s.magnetismReturn);
            }
            else if (key == "agentBaseNameToDuration") {
                val.convert(s.agentBaseNameToDuration);
            }
            else if (key == "field") {
                val.convert(s.field);
            }
            else if (key == "lastCount") {
                val.convert(s.lastCount);
            }
            else if (key == "value") {
                val.convert(s.value);
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::process::MagneticField>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::MagneticField& v) const
    {
        const auto& s = v.state();

        o.pack_map(8);

        o.pack("rng");
        o.pack(s.rng);

        o.pack("logReturn");
        o.pack(s.logReturn);

        o.pack("magnetism");
        o.pack(s.magnetism);

        o.pack("magnetismReturn");
        o.pack(s.magnetismReturn);

        o.pack("agentBaseNameToDuration");
        o.pack(s.agentBaseNameToDuration);

        o.pack("field");
        o.pack(s.field);

        o.pack("lastCount");
        o.pack(s.lastCount);

        o.pack("value");
        o.pack(s.value);

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------