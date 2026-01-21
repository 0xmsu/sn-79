/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <taosim/message/PayloadFactory.hpp>

//-------------------------------------------------------------------------

void BookStateMessagePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        if (json.Parse(bookStateJsonStr.c_str()).HasParseError()) {
            throw std::runtime_error(fmt::format(
                "{}: Book state ill-formed Json :\n{}",
                std::source_location::current().function_name(),
                bookStateJsonStr));
        }
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

BookStateMessagePayload::Ptr BookStateMessagePayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<BookStateMessagePayload>(json);
}

//-------------------------------------------------------------------------

void DistributedAgentResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
        payload->jsonSerialize(json, "payload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

DistributedAgentResponsePayload::Ptr DistributedAgentResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<DistributedAgentResponsePayload>(
        json["agentId"].GetInt(),
        PayloadFactory::createFromJsonMessage(json));
}

//-------------------------------------------------------------------------
