/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ClosePosition.hpp"

//-------------------------------------------------------------------------

void ClosePosition::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("event", rapidjson::Value{"close", allocator}, allocator);
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
        taosim::json::setOptionalMember(json, "volume", volume.transform(&taosim::util::decimal2double));
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

ClosePosition::Ptr ClosePosition::fromJson(const rapidjson::Value& json)
{
    return ClosePosition::Ptr{new ClosePosition(
        json["orderId"].GetUint64(),
        !json["volume"].IsNull()
            ? std::make_optional(taosim::json::getDecimal(json["volume"]))
            : std::nullopt)};
}

//-------------------------------------------------------------------------
