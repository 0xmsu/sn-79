/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "JsonSerializable.hpp"
#include "Order.hpp"
#include "common.hpp"
#include "json_util.hpp"

#include <memory>
#include <optional>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

struct ClosePosition : public JsonSerializable
{
    using Ptr = std::shared_ptr<ClosePosition>;

    OrderID id;
    std::optional<taosim::decimal_t> volume;

    ClosePosition() = default;

    ClosePosition(OrderID id, std::optional<taosim::decimal_t> volume = {}) noexcept
        : id{id}, volume{volume}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);
};

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<ClosePosition>
{
    const msgpack::object& operator()(const msgpack::object& o, ClosePosition& v) const
    {
        if (o.type != msgpack::type::MAP) {
            taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "orderId") {
                v.id = val.as<OrderID>();
            }
            else if (key == "volume") {
                v.volume = val.as<std::optional<taosim::decimal_t>>();
            }
        }

        return o;
    }
};

template<>
struct pack<ClosePosition>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const ClosePosition& v) const
    {
        o.pack_map(3);

        o.pack("event");
        o.pack("close");

        o.pack("orderId");
        o.pack(v.id);

        o.pack("volume");
        o.pack(v.volume);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
