/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/Message.hpp>

#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>

//-------------------------------------------------------------------------

Message::Message(
    Timestamp occurrence,
    Timestamp arrival,
    const std::string& source,
    const std::string& target,
    const std::string& type,
    MessagePayload::Ptr payload) noexcept
    : occurrence{occurrence}, arrival{arrival}, source{source}, type{type}, payload{payload}
{
    targets = target | views::split(s_targetDelim) | ranges::to<decltype(targets)>;
}

//-------------------------------------------------------------------------

void Message::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("timestamp", rapidjson::Value{occurrence}, allocator);
        json.AddMember("delay", rapidjson::Value{arrival - occurrence}, allocator);
        json.AddMember("source", rapidjson::Value{source.c_str(), allocator}, allocator);
        json.AddMember(
            "target",
            rapidjson::Value{
                fmt::format("{}", fmt::join(targets, std::string{1, s_targetDelim})).c_str(),
                allocator},
            allocator);
        json.AddMember("type", rapidjson::Value{type.c_str(), allocator}, allocator);
        payload->jsonSerialize(json, "payload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

Message::Ptr Message::fromJsonMessage(const rapidjson::Value& json) noexcept
{
    return Message::create(
        Timestamp{json["timestamp"].GetUint64()},
        Timestamp{json["timestamp"].GetUint64() + json["delay"].GetUint64()},
        json["source"].GetString(),
        json["target"].GetString(),
        json["type"].GetString(),
        PayloadFactory::createFromJsonMessage(json));
}

//-------------------------------------------------------------------------

Message::Ptr Message::fromJsonResponse(
    const rapidjson::Value& json, Timestamp timestamp, const std::string& source) noexcept
{
    return Message::create(
        timestamp,
        timestamp + Timestamp{json["delay"].GetUint64()},
        source,
        "EXCHANGE",
        fmt::format("{}_{}", "DISTRIBUTED", json["type"].GetString()),
        DistributedAgentResponsePayload::fromJson(json));
}

//-------------------------------------------------------------------------
