/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/event/OrderEvent.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

void OrderEvent::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
        json.AddMember("timestamp", rapidjson::Value{timestamp}, allocator);
        json.AddMember(
            "volume", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember(
            "leverage", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        json.AddMember(
            "direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
        if (price) {
            json.AddMember("price", taosim::util::decimal2double(*price), allocator);
        } else {
            json.AddMember("price", rapidjson::Value{}.SetNull(), allocator);
        }
        if (postOnly) {
            json.AddMember("postOnly", rapidjson::Value{*postOnly}, allocator);
        }
        if (timeInForce) {
            json.AddMember(
                "timeInForce",
                rapidjson::Value{magic_enum::enum_name(*timeInForce).data(), allocator},
                allocator);
        }
        if (expiryPeriod) {
            taosim::json::setOptionalMember(json, "expiryPeriod", *expiryPeriod);
        }
        json.AddMember("event", rapidjson::Value{"place", allocator}, allocator);
        json.AddMember("agentId", rapidjson::Value{ctx.agentId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", ctx.clientOrderId);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------