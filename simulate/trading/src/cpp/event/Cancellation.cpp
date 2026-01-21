/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/event/Cancellation.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

void Cancellation::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("e", rapidjson::Value{"cancel", allocator}, allocator);
        json.AddMember("i", rapidjson::Value{id}, allocator);
        taosim::json::setOptionalMember(json, "v", volume.transform(&taosim::util::decimal2double));
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void Cancellation::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("event", rapidjson::Value{"cancel", allocator}, allocator);
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
        taosim::json::setOptionalMember(json, "volume", volume.transform(&taosim::util::decimal2double));
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

Cancellation::Ptr Cancellation::fromJson(const rapidjson::Value& json)
{
    return Cancellation::Ptr{new Cancellation(
        json["orderId"].GetUint64(),
        !json["volume"].IsNull()
            ? std::make_optional(taosim::json::getDecimal(json["volume"]))
            : std::nullopt)};
}

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------