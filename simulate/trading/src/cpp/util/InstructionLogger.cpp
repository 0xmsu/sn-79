/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "InstructionLogger.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

//-------------------------------------------------------------------------

namespace taosim
{

//-------------------------------------------------------------------------

void InstructionLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        std::visit(
            [&](auto&& pld) {
                using T = std::remove_cvref_t<decltype(pld)>::element_type;
                static_assert(
                    requires(T t, rapidjson::Document& j, const std::string& k) {
                        t.L3Serialize(j, k);
                    },
                    "InstructionLogContext::payload should be L3Serializable");
                pld->L3Serialize(json, "in");
            },
            payload);
        json.AddMember("a", rapidjson::Value{agentId}, allocator);
        json.AddMember("i", rapidjson::Value{orderId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void InstructionLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        std::visit(
            [&](auto&& pld) {
                using T = std::remove_cvref_t<decltype(pld)>::element_type;
                static_assert(json::IsJsonSerializable<T>);
                pld->jsonSerialize(json, "instruction");
            },
            payload);
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
        json.AddMember("orderId", rapidjson::Value{orderId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim

//-------------------------------------------------------------------------
