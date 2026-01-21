/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/event/CancellationEvent.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

void CancellationEvent::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        auto& allocator = json.GetAllocator();
        cancellation.jsonSerialize(json);
        json.AddMember("timestamp", rapidjson::Value{timestamp}, allocator);
        json.AddMember("price", rapidjson::Value{taosim::util::decimal2double(price)}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------