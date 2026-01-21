/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/Fees.hpp>
#include "JsonSerializable.hpp"
#include "Order.hpp"
#include "Timestamp.hpp"
#include "common.hpp"
#include <taosim/mp/mp.hpp>
#include "util.hpp"

#include <memory>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

using TradeID = uint32_t;

//-------------------------------------------------------------------------

struct Trade : public JsonSerializable
{
    using Ptr = std::shared_ptr<Trade>;

    Trade() noexcept = default;

    Trade(
        TradeID id,
        Timestamp timestamp,
        OrderDirection direction,
        OrderID aggressingOrderID,
        OrderID restingOrderID,
        taosim::decimal_t volume,
        taosim::decimal_t price) noexcept;

    [[nodiscard]] TradeID id() const noexcept { return m_id; }
    [[nodiscard]] OrderDirection direction() const noexcept { return m_direction; }
    [[nodiscard]] Timestamp timestamp() const noexcept { return m_timestamp; }
    [[nodiscard]] OrderID aggressingOrderID() const noexcept { return m_aggressingOrderID; }
    [[nodiscard]] OrderID restingOrderID() const noexcept { return m_restingOrderID; }
    [[nodiscard]] taosim::decimal_t volume() const noexcept { return m_volume; }
    [[nodiscard]] taosim::decimal_t price() const noexcept { return m_price; }

    void setTimestamp(Timestamp timestamp) noexcept { m_timestamp = timestamp; }

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<Trade, Args...> && taosim::mp::IsPointer<typename Trade::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return Trade::Ptr{new Trade(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    TradeID m_id;
    Timestamp m_timestamp;
    OrderDirection m_direction;
    OrderID m_aggressingOrderID;
    OrderID m_restingOrderID;
    taosim::decimal_t m_volume;
    taosim::decimal_t m_price;

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("tradeId", m_id),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("aggressingOrderId", m_aggressingOrderID),
        MSGPACK_NVP("restingOrderId", m_restingOrderID),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("price", m_price));
};

//-------------------------------------------------------------------------

struct TradeContext : public JsonSerializable
{
    BookId bookId;
    AgentId aggressingAgentId;
    AgentId restingAgentId;
    taosim::exchange::Fees fees;

    TradeContext() = default;

    TradeContext(
        BookId bookId,
        AgentId aggressingAgentId,
        AgentId restingAgentId,
        taosim::exchange::Fees fees) noexcept
        : bookId{bookId},
          aggressingAgentId{aggressingAgentId},
          restingAgentId{restingAgentId},
          fees{fees}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static TradeContext fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId, aggressingAgentId, restingAgentId, fees);
};

//-------------------------------------------------------------------------

struct TradeLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<TradeLogContext>;

    AgentId aggressingAgentId;
    AgentId restingAgentId;
    BookId bookId;
    taosim::exchange::Fees fees;

    TradeLogContext() noexcept = default;

    TradeLogContext(
        AgentId aggressingAgentId,
        AgentId restingAgentId,
        BookId bookId,
        taosim::exchange::Fees fees) noexcept
        : aggressingAgentId{aggressingAgentId},
          restingAgentId{restingAgentId},
          bookId{bookId},
          fees{fees}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<TradeLogContext, Args...>
        && taosim::mp::IsPointer<typename TradeLogContext::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return TradeLogContext::Ptr{new TradeLogContext(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId, aggressingAgentId, restingAgentId, fees);
};

//-------------------------------------------------------------------------

struct TradeWithLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<TradeWithLogContext>;

    Trade::Ptr trade;
    TradeLogContext::Ptr logContext;

    TradeWithLogContext() noexcept = default;

    TradeWithLogContext(Trade::Ptr trade, TradeLogContext::Ptr logContext) noexcept
        : trade{trade}, logContext{logContext}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<TradeWithLogContext, Args...>
        && taosim::mp::IsPointer<typename Trade::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return TradeWithLogContext::Ptr{new TradeWithLogContext(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(trade, logContext);
};

//-------------------------------------------------------------------------
