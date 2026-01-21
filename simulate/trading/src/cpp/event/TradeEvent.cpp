/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/event/TradeEvent.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

void TradeEvent::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        trade->jsonSerialize(json);
        auto& allocator = json.GetAllocator();
        json.AddMember("event", rapidjson::Value{"trade", allocator}, allocator);
        json.AddMember("aggressingAgentId", rapidjson::Value{ctx.aggressingAgentId}, allocator);
        json.AddMember("restingAgentId", rapidjson::Value{ctx.restingAgentId}, allocator);
        taosim::json::serializeHelper(
            json,
            "fees",
            [this](rapidjson::Document& json) {
                json.SetObject();
                auto& allocator = json.GetAllocator();
                json.AddMember(
                    "maker", rapidjson::Value{taosim::util::decimal2double(ctx.fees.maker)}, allocator);
                json.AddMember(
                    "taker", rapidjson::Value{taosim::util::decimal2double(ctx.fees.taker)}, allocator);
            });
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------