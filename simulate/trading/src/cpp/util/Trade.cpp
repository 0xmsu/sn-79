/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Trade.hpp"

#include "Order.hpp"
#include "util.hpp"

#include <iostream>

//-------------------------------------------------------------------------

Trade::Trade(
    TradeID id,
    Timestamp timestamp,
    OrderDirection direction,
    OrderID aggressingOrderID,
    OrderID restingOrderID,
    taosim::decimal_t volume,
    taosim::decimal_t price) noexcept
    : m_id{id},
      m_timestamp{timestamp},
      m_direction{direction},
      m_aggressingOrderID{aggressingOrderID},
      m_restingOrderID{restingOrderID},
      m_volume{volume},
      m_price{price}
{}

//-------------------------------------------------------------------------

void Trade::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("m", rapidjson::Value{m_id}, allocator);
        json.AddMember("j", rapidjson::Value{m_timestamp}, allocator);
        json.AddMember("d", rapidjson::Value{std::to_underlying(m_direction)}, allocator);
        json.AddMember("ai", rapidjson::Value{m_aggressingOrderID}, allocator);
        json.AddMember("ri", rapidjson::Value{m_restingOrderID}, allocator);
        json.AddMember("v", rapidjson::Value{taosim::util::decimal2double(m_volume)}, allocator);
        json.AddMember("p", rapidjson::Value{taosim::util::decimal2double(m_price)}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void Trade::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("tradeId", rapidjson::Value{m_id}, allocator);
        json.AddMember("timestamp", rapidjson::Value{m_timestamp}, allocator);
        json.AddMember("direction", rapidjson::Value{std::to_underlying(m_direction)}, allocator);
        json.AddMember("aggressingOrderId", rapidjson::Value{m_aggressingOrderID}, allocator);
        json.AddMember("restingOrderId", rapidjson::Value{m_restingOrderID}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::decimal2double(m_volume)}, allocator);
        json.AddMember("price", rapidjson::Value{taosim::util::decimal2double(m_price)}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

Trade::Ptr Trade::fromJson(const rapidjson::Value& json)
{
    return Trade::create(
        json["tradeId"].GetUint(),
        json["timestamp"].GetUint64(),
        OrderDirection{json["direction"].GetUint()},
        json["aggressingOrderId"].GetUint64(),
        json["restingOrderId"].GetUint64(),
        taosim::json::getDecimal(json["volume"]),
        taosim::json::getDecimal(json["price"]));
}

//-------------------------------------------------------------------------

void TradeContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("aggressingAgentId", rapidjson::Value{aggressingAgentId}, allocator);
        json.AddMember("restingAgentId", rapidjson::Value{restingAgentId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::serializeHelper(
            json,
            "fees",
            [this](rapidjson::Document& json) {
                json.SetObject();
                auto& allocator = json.GetAllocator();
                json.AddMember(
                    "maker", rapidjson::Value{taosim::util::decimal2double(fees.maker)}, allocator);
                json.AddMember(
                    "taker", rapidjson::Value{taosim::util::decimal2double(fees.taker)}, allocator);
            });
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

TradeContext TradeContext::fromJson(const rapidjson::Value& json)
{
    return TradeContext(
        json["aggressingAgentId"].GetInt(),
        json["restingAgentId"].GetInt(),
        json["bookId"].GetUint(),
        taosim::exchange::Fees{
            .maker = taosim::json::getDecimal(json["fees"]["maker"]),
            .taker = taosim::json::getDecimal(json["fees"]["taker"])});
}

//-------------------------------------------------------------------------

void TradeLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("aa", rapidjson::Value{aggressingAgentId}, allocator);
        json.AddMember("ra", rapidjson::Value{restingAgentId}, allocator);
        json.AddMember("b", rapidjson::Value{bookId}, allocator);
        taosim::json::serializeHelper(
            json,
            "fs",
            [this](rapidjson::Document& json) {
                json.SetObject();
                auto& allocator = json.GetAllocator();
                json.AddMember(
                    "mk", rapidjson::Value{taosim::util::decimal2double(fees.maker)}, allocator);
                json.AddMember(
                    "tk", rapidjson::Value{taosim::util::decimal2double(fees.taker)}, allocator);
            });
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void TradeLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("aggressingAgentId", rapidjson::Value{aggressingAgentId}, allocator);
        json.AddMember("restingAgentId", rapidjson::Value{restingAgentId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::serializeHelper(
            json,
            "fees",
            [this](rapidjson::Document& json) {
                json.SetObject();
                auto& allocator = json.GetAllocator();
                json.AddMember(
                    "maker", rapidjson::Value{taosim::util::decimal2double(fees.maker)}, allocator);
                json.AddMember(
                    "taker", rapidjson::Value{taosim::util::decimal2double(fees.taker)}, allocator);
            });
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

TradeLogContext::Ptr TradeLogContext::fromJson(const rapidjson::Value& json)
{
    return TradeLogContext::create(
        json["aggressingAgentId"].GetInt(),
        json["restingAgentId"].GetInt(),
        json["bookId"].GetUint(),
        taosim::exchange::Fees{
            .maker = taosim::json::getDecimal(json["fees"]["maker"]),
            .taker = taosim::json::getDecimal(json["fees"]["taker"])});
}

//-------------------------------------------------------------------------

void TradeWithLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        trade->L3Serialize(json, "t");
        logContext->L3Serialize(json, "g");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void TradeWithLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        trade->jsonSerialize(json, "trade");
        logContext->jsonSerialize(json, "logContext");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

TradeWithLogContext::Ptr TradeWithLogContext::fromJson(const rapidjson::Value& json)
{
    return TradeWithLogContext::create(
        Trade::fromJson(json["trade"]),
        TradeLogContext::fromJson(json["logContext"]));
}

//-------------------------------------------------------------------------