/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/ALGOTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/common.hpp>
#include <taosim/dsa/serialization/PriorityQueue.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

MSGPACK_ADD_ENUM(taosim::agent::ALGOTraderStatus);

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::TimestampedVolume>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::TimestampedVolume& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "timestamp") {
                v.timestamp = val.as<Timestamp>();
            }
            else if (key == "volume") {
                v.volume = val.as<taosim::decimal_t>();
            }
            else if (key == "price") {
                v.price = val.as<taosim::decimal_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::TimestampedVolume>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::TimestampedVolume& v) const
    {
        o.pack_map(3);

        o.pack("timestamp");
        o.pack(v.timestamp);

        o.pack("volume");
        o.pack(v.volume);

        o.pack("price");
        o.pack(v.price);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::BookStat>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::BookStat& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "bid") {
                v.bid = val.as<double>();
            }
            else if (key == "ask") {
                v.ask = val.as<double>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::BookStat>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::BookStat& v) const
    {
        o.pack_map(2);

        o.pack("bid");
        o.pack(v.bid);

        o.pack("ask");
        o.pack(v.ask);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::ALGOTraderVolumeStats>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::ALGOTraderVolumeStats& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "queue") {
                using T = std::remove_cvref_t<decltype(v.queue())>;
                v.queue() = T(T::CompareType{}, val.as<T::container_type>());
            }
            else if (key == "rollingSum") {
                v.rollingSum() = val.as<taosim::decimal_t>();
            }
            else if (key == "priceHistory") {
                using T = std::remove_cvref_t<decltype(v.priceHistory())>;
                v.priceHistory() = val.as<T>();
            }
            else if (key == "logRets") {
                using T = std::remove_cvref_t<decltype(v.logRets())>;
                v.logRets() = val.as<T>();
            }
            else if (key == "priceLast") {
                v.priceLast() = val.as<double>();
            }
            else if (key == "variance") {
                v.variance() = val.as<double>();
            }
            else if (key == "estimatedVol") {
                v.estimatedVol() = val.as<double>();
            }
            else if (key == "lastSeq") {
                v.lastSeq() = val.as<Timestamp>();
            }
            else if (key == "bookSlopes") {
                using T = std::remove_cvref_t<decltype(v.bookSlopes())>;
                v.bookSlopes() = val.as<T>();
            }
            else if (key == "bookVolumes") {
                using T = std::remove_cvref_t<decltype(v.bookVolumes())>;
                v.bookVolumes() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::ALGOTraderVolumeStats>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::ALGOTraderVolumeStats& v) const
    {
        o.pack_map(10);

        o.pack("queue");
        o.pack(v.queue());

        o.pack("rollingSum");
        o.pack(v.rollingSum());

        o.pack("priceHistory");
        o.pack(v.priceHistory());

        o.pack("logRets");
        o.pack(v.logRets());

        o.pack("priceLast");
        o.pack(v.priceLast());

        o.pack("variance");
        o.pack(v.variance());

        o.pack("estimatedVol");
        o.pack(v.estimatedVol());

        o.pack("lastSeq");
        o.pack(v.lastSeq());

        o.pack("bookSlopes");
        o.pack(v.bookSlopes());

        o.pack("bookVolumes");
        o.pack(v.bookVolumes());

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::ALGOTraderState>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::ALGOTraderState& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "status") {
                v.status = val.as<taosim::agent::ALGOTraderStatus>();
            }
            else if (key == "marketFeedLatency") {
                v.marketFeedLatency = val.as<Timestamp>();
            }
            else if (key == "volumeStats") {
                v.volumeStats = val.as<taosim::agent::ALGOTraderVolumeStats>();
            }
            else if (key == "volumeToBeExecuted") {
                v.volumeToBeExecuted = val.as<taosim::decimal_t>();
            }
            else if (key == "direction") {
                v.direction = val.as<OrderDirection>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::ALGOTraderState>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::ALGOTraderState& v) const
    {
        o.pack_map(5);

        o.pack("status");
        o.pack(v.status);

        o.pack("marketFeedLatency");
        o.pack(v.marketFeedLatency);

        o.pack("volumeStats");
        o.pack(v.volumeStats);

        o.pack("volumeToBeExecuted");
        o.pack(v.volumeToBeExecuted);

        o.pack("direction");
        o.pack(v.direction);

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::ALGOTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::ALGOTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "state") {
                using T = std::remove_cvref_t<decltype(v.state())>;
                v.state() = val.as<T>();
            }
            else if (key == "lastPrice") {
                using T = std::remove_cvref_t<decltype(v.lastPrice())>;
                v.lastPrice() = val.as<T>();
            }
            else if (key == "topLevel") {
                using T = std::remove_cvref_t<decltype(v.topLevel())>;
                v.topLevel() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::ALGOTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::ALGOTraderAgent& v) const
    {
        o.pack_map(3);

        o.pack("state");
        o.pack(v.state());

        o.pack("lastPrice");
        o.pack(v.lastPrice());

        o.pack("topLevel");
        o.pack(v.topLevel());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
