/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Order.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

#include "util.hpp"

//-------------------------------------------------------------------------

void BasicOrder::removeVolume(taosim::decimal_t decrease)
{
    if (decrease > m_volume) {
        throw std::runtime_error(fmt::format(
            "{}: Volume to be removed ({}) is greater than standing volume ({})",
            std::source_location::current().function_name(),
            decrease,
            m_volume));
    }
    m_volume -= decrease;
}

//-------------------------------------------------------------------------

void BasicOrder::removeLeveragedVolume(taosim::decimal_t decrease)
{
    if (m_leverage == 0_dec){
        removeVolume(decrease);
    } else{
        taosim::decimal_t leveragedVolume = m_volume * (1_dec + m_leverage);
        if (decrease > leveragedVolume) {
            throw std::runtime_error(fmt::format(
                "{}: Volume to be removed ({}) is greater than standing volume ({})",
                std::source_location::current().function_name(),
                decrease,
                leveragedVolume));
        }
        m_volume -= decrease / (1_dec + m_leverage);
    }
}

//-------------------------------------------------------------------------

void BasicOrder::setVolume(taosim::decimal_t newVolume)
{
    if (newVolume < 0_dec) {
        throw std::invalid_argument(fmt::format(
            "{}: Negative volume ({})",
            std::source_location::current().function_name(),
            newVolume));
    }
    m_volume = newVolume;
}

//-------------------------------------------------------------------------

void BasicOrder::setLeverage(taosim::decimal_t newLeverage)
{
    if (newLeverage < 0_dec) {
        throw std::invalid_argument(fmt::format(
            "{}: Negative leverage ({})",
            std::source_location::current().function_name(),
            newLeverage));
    }
    m_leverage = newLeverage;
}

//-------------------------------------------------------------------------

void BasicOrder::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{m_id}, allocator);
        json.AddMember("timestamp", rapidjson::Value{m_timestamp}, allocator);
        json.AddMember(
            "volume", rapidjson::Value{taosim::util::decimal2double(m_volume)}, allocator);
        json.AddMember(
            "leverage", rapidjson::Value{taosim::util::decimal2double(m_leverage)}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

BasicOrder::BasicOrder(
    OrderID id,
    Timestamp timestamp,
    taosim::decimal_t volume,
    taosim::decimal_t leverage) noexcept
    : m_id{id},
      m_timestamp{timestamp},
      m_volume{volume},
      m_leverage{leverage}    
{}

//-------------------------------------------------------------------------

Order::Order(
    OrderID orderId,
    Timestamp timestamp,
    taosim::decimal_t volume,
    OrderDirection direction,
    taosim::decimal_t leverage,
    STPFlag stpFlag,
    SettleFlag settleFlag,
    Currency currency) noexcept
    : BasicOrder(orderId, timestamp, volume, leverage),
      m_direction{direction}, m_stpFlag{stpFlag}, m_settleFlag{settleFlag}, m_currency{currency}
{}

//-------------------------------------------------------------------------

void Order::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        BasicOrder::jsonSerialize(json);
        auto& allocator = json.GetAllocator();
        json.AddMember(
            "direction", rapidjson::Value{std::to_underlying(m_direction)}, allocator);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(m_stpFlag).data(), allocator},
            allocator);
        std::visit(
            [&](auto&& flag) {
                using T = std::remove_cvref_t<decltype(flag)>;
                if constexpr (std::same_as<T, SettleType>) {
                    json.AddMember(
                        "settleFlag",
                        rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
                        allocator);
                } else if constexpr (std::same_as<T, OrderID>) {
                    json.AddMember("settleFlag", rapidjson::Value{flag}, allocator);
                }
            }, m_settleFlag);
        json.AddMember(
            "currency",
            rapidjson::Value{magic_enum::enum_name(m_currency).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

MarketOrder::MarketOrder(
    OrderID orderId,
    Timestamp timestamp,
    taosim::decimal_t volume,
    OrderDirection direction,
    taosim::decimal_t leverage,
    STPFlag stpFlag,
    SettleFlag settleFlag,
    Currency currency) noexcept
    : Order(orderId, timestamp, volume, direction, leverage, stpFlag, settleFlag, currency)
{}

//-------------------------------------------------------------------------

void MarketOrder::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("i", rapidjson::Value{id()}, allocator);
        json.AddMember("j", rapidjson::Value{timestamp()}, allocator);
        json.AddMember("v", rapidjson::Value{taosim::util::decimal2double(volume())}, allocator);
        json.AddMember("d", rapidjson::Value{std::to_underlying(direction())}, allocator);
        json.AddMember("l", rapidjson::Value{taosim::util::decimal2double(leverage())}, allocator);
        json.AddMember(
            "s", rapidjson::Value{magic_enum::enum_name(stpFlag()).data(), allocator}, allocator);
        std::visit(
            [&](auto&& flag) {
                using T = std::remove_cvref_t<decltype(flag)>;
                if constexpr (std::same_as<T, SettleType>) {
                    json.AddMember(
                        "f",
                        rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
                        allocator);
                } else if constexpr (std::same_as<T, OrderID>) {
                    json.AddMember("f", rapidjson::Value{flag}, allocator);
                } else {
                    static_assert(false, "Non-exhaustive visitor");
                }
            },
            settleFlag());
        json.AddMember(
            "n",
            rapidjson::Value{magic_enum::enum_name(currency()).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void MarketOrder::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        Order::jsonSerialize(json);
        auto& allocator = json.GetAllocator();
        json.AddMember("price", rapidjson::Value{}.SetNull(), allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

MarketOrder::Ptr MarketOrder::fromJson(const rapidjson::Value& json)
{
    return Ptr{new MarketOrder(
        json["orderId"].GetUint64(),
        json["timestamp"].GetUint64(),
        taosim::json::getDecimal(json["volume"]),
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["leverage"]),
        STPFlag{json["stpFlag"].GetUint()},
        json["settleFlag"].IsInt() && magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).has_value()
            ? SettleFlag(magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).value())
            : SettleFlag(static_cast<OrderID>(json["settleFlag"].GetUint())),
        magic_enum::enum_cast<Currency>(json["currency"].GetInt()).value_or(Currency::BASE)
    )};
}

//-------------------------------------------------------------------------

LimitOrder::LimitOrder(
    OrderID orderId,
    Timestamp timestamp,
    taosim::decimal_t volume,
    OrderDirection direction,
    taosim::decimal_t price,
    taosim::decimal_t leverage,
    STPFlag stpFlag,
    SettleFlag settleFlag,
    bool postOnly,
    taosim::TimeInForce timeInForce,
    std::optional<Timestamp> expiryPeriod,
    Currency currency) noexcept
    : Order(orderId, timestamp, volume, direction, leverage, stpFlag, settleFlag, currency),
      m_price{price},
      m_postOnly{postOnly},
      m_timeInForce{timeInForce},
      m_expiryPeriod{expiryPeriod}
{}

//-------------------------------------------------------------------------

void LimitOrder::setPrice(taosim::decimal_t newPrice)
{
    if (newPrice <= 0_dec) {
        throw std::invalid_argument(fmt::format(
            "{}: Non-positive price ({})",
            std::source_location::current().function_name(),
            newPrice));
    }
    m_price = newPrice;
}

//-------------------------------------------------------------------------

void LimitOrder::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("i", rapidjson::Value{id()}, allocator);
        json.AddMember("j", rapidjson::Value{timestamp()}, allocator);
        json.AddMember("v", rapidjson::Value{taosim::util::decimal2double(volume())}, allocator);
        json.AddMember("d", rapidjson::Value{std::to_underlying(direction())}, allocator);
        json.AddMember("l", rapidjson::Value{taosim::util::decimal2double(leverage())}, allocator);
        json.AddMember(
            "s", rapidjson::Value{magic_enum::enum_name(stpFlag()).data(), allocator}, allocator);
        std::visit(
            [&](auto&& flag) {
                using T = std::remove_cvref_t<decltype(flag)>;
                if constexpr (std::same_as<T, SettleType>) {
                    json.AddMember(
                        "f",
                        rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
                        allocator);
                } else if constexpr (std::same_as<T, OrderID>) {
                    json.AddMember("f", rapidjson::Value{flag}, allocator);
                } else {
                    static_assert(false, "Non-exhaustive visitor");
                }
            },
            settleFlag());
        json.AddMember(
            "n",
            rapidjson::Value{magic_enum::enum_name(currency()).data(), allocator},
            allocator);
        json.AddMember("p", rapidjson::Value{taosim::util::decimal2double(m_price)}, allocator);
        json.AddMember("y", rapidjson::Value{m_postOnly}, allocator);
        json.AddMember(
            "r",
            rapidjson::Value{magic_enum::enum_name(m_timeInForce).data(), allocator},
            allocator);
        taosim::json::setOptionalMember(json, "x", m_expiryPeriod);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void LimitOrder::jsonSerialize(rapidjson::Document &json, const std::string &key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        Order::jsonSerialize(json);
        auto& allocator = json.GetAllocator();
        json.AddMember("price", rapidjson::Value{taosim::util::decimal2double(m_price)}, allocator);
        json.AddMember("postOnly", rapidjson::Value{m_postOnly}, allocator);
        json.AddMember(
            "timeInForce",
            rapidjson::Value{magic_enum::enum_name(m_timeInForce).data(), allocator},
            allocator);
        taosim::json::setOptionalMember(json, "expiryPeriod", m_expiryPeriod);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

LimitOrder::Ptr LimitOrder::fromJson(
    const rapidjson::Value& json, [[maybe_unused]] int priceDecimals, int volumeDecimals)
{
    return Ptr{new LimitOrder(
        json["orderId"].GetUint64(),
        json["timestamp"].GetUint64(),
        taosim::util::round(taosim::json::getDecimal(json["volume"]), volumeDecimals),
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["price"]),
        taosim::json::getDecimal(json["leverage"]),
        STPFlag{json["stpFlag"].GetUint()},
        json["settleFlag"].IsInt() && magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).has_value()
            ? SettleFlag(magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).value())
            : SettleFlag(static_cast<OrderID>(json["settleFlag"].GetUint()))
    )};
}

//-------------------------------------------------------------------------

OrderClientContext OrderClientContext::fromJson(const rapidjson::Value& json)
{
    return OrderClientContext{
        json["agentId"].GetInt(),
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt};
}

//-------------------------------------------------------------------------

void OrderContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

OrderContext OrderContext::fromJson(const rapidjson::Value& json)
{
    return OrderContext(
        json["agentId"].GetInt(),
        json["bookId"].GetUint(),
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint64()) 
            : std::nullopt);
}

//-------------------------------------------------------------------------

void OrderLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("a", rapidjson::Value{agentId}, allocator);
        json.AddMember("b", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void OrderLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void OrderWithLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        if (const auto o = std::dynamic_pointer_cast<MarketOrder>(order)) {
            o->L3Serialize(json, "o");
        } else if (const auto o = std::dynamic_pointer_cast<LimitOrder>(order)) {
            o->L3Serialize(json, "o");
        }
        logContext->L3Serialize(json, "g");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void OrderWithLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        order->jsonSerialize(json, "order");
        logContext->jsonSerialize(json, "logContext");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------
