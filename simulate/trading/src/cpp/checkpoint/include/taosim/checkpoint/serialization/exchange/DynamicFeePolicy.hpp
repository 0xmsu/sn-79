/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/DynamicFeePolicy.hpp>
#include <taosim/serialization/msgpack/boost/circular_buffer.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::exchange::Volumes>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::Volumes& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "aggressive") {
                v.aggressive = val.as<taosim::decimal_t>();
            }
            else if (key == "passive") {
                v.passive = val.as<taosim::decimal_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::exchange::Volumes>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::Volumes& v) const
    {
        o.pack_map(2);

        o.pack("aggressive");
        o.pack(v.aggressive);

        o.pack("passive");
        o.pack(v.passive);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::exchange::DynamicFeePolicy>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::exchange::DynamicFeePolicy& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "lastUpdate") {
                v.lastUpdate() = val.as<Timestamp>();
            }
            else if (key == "totalVolumes") {
                using T = std::remove_cvref_t<decltype(v.totalVolumes())>;
                v.totalVolumes() = val.as<T>();
            }
            else if (key == "totalVolumesPrev") {
                using T = std::remove_cvref_t<decltype(v.totalVolumesPrev())>;
                v.totalVolumesPrev() = val.as<T>();
            }
            else if (key == "volumes") {
                using T = std::remove_cvref_t<decltype(v.volumes())>;
                v.volumes() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::exchange::DynamicFeePolicy>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::exchange::DynamicFeePolicy& v) const
    {
        o.pack_map(5);

        o.pack("type");
        o.pack("dynamic");

        o.pack("lastUpdate");
        o.pack(v.lastUpdate());

        o.pack("totalVolumes");
        o.pack(v.totalVolumes());

        o.pack("totalVolumesPrev");
        o.pack(v.totalVolumesPrev());

        o.pack("volumes");
        o.pack(v.volumes());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
