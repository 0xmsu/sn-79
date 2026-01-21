/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/serialization/decimal.hpp>
#include "CheckpointSerializable.hpp"
#include "JsonSerializable.hpp"
#include "common.hpp"
#include "Flags.hpp"

#include <msgpack.hpp>

//-------------------------------------------------------------------------

using ClientOrderID = std::decay_t<OrderID>;
using taosim::STPFlag;
using taosim::SettleFlag;
using taosim::SettleType;

enum class OrderDirection : uint32_t
{
    BUY,
    SELL
};

MSGPACK_ADD_ENUM(OrderDirection);

enum class Currency : uint32_t
{
    BASE,
    QUOTE    
};

MSGPACK_ADD_ENUM(Currency);

[[nodiscard]] constexpr std::string_view OrderDirection2StrView(OrderDirection dir) noexcept
{
    return magic_enum::enum_name(dir);
}

template<>
struct fmt::formatter<OrderDirection>
{
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(OrderDirection dir, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", OrderDirection2StrView(dir));
    }
};

enum class OrderErrorCode : uint32_t
{
    VALID,
    NONEXISTENT_ACCOUNT,
    INSUFFICIENT_BASE,
    INSUFFICIENT_QUOTE,
    EMPTY_BOOK,
    PRICE_INCREMENT_VIOLATED,
    VOLUME_INCREMENT_VIOLATED,
    EXCEEDING_LOAN,
    CONTRACT_VIOLATION,
    INVALID_LEVERAGE,
    INVALID_VOLUME,
    INVALID_PRICE,
    EXCEEDING_MAX_ORDERS,
    DUAL_POSITION,
    MINIMUM_ORDER_SIZE_VIOLATION
};

[[nodiscard]] constexpr std::string_view OrderErrorCode2StrView(OrderErrorCode ec) noexcept
{
    return magic_enum::enum_name(ec);
}

template<>
struct fmt::formatter<OrderErrorCode>
{
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(OrderErrorCode ec, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", OrderErrorCode2StrView(ec));
    }
};

//-------------------------------------------------------------------------

struct BasicOrder : public JsonSerializable
{
    BasicOrder() noexcept = default;

    BasicOrder(
        OrderID id,
        Timestamp timestamp,
        taosim::decimal_t volume,
        taosim::decimal_t leverage = 0_dec) noexcept;

    virtual ~BasicOrder() noexcept = default;

    [[nodiscard]] OrderID id() const noexcept { return m_id; }
    [[nodiscard]] Timestamp timestamp() const noexcept { return m_timestamp; }
    [[nodiscard]] taosim::decimal_t volume() const noexcept { return m_volume; }
    [[nodiscard]] taosim::decimal_t totalVolume() const noexcept { return m_volume * taosim::util::dec1p(m_leverage); }
    [[nodiscard]] taosim::decimal_t leverage() const noexcept { return m_leverage; }
    
    void removeVolume(taosim::decimal_t decrease);
    void removeLeveragedVolume(taosim::decimal_t decrease);
    void setVolume(taosim::decimal_t newVolume);
    void setLeverage(taosim::decimal_t newLeverage);

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    OrderID m_id;
    Timestamp m_timestamp;
    taosim::decimal_t m_volume;
    taosim::decimal_t m_leverage{};

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage));
};

//-------------------------------------------------------------------------

struct Order : public BasicOrder
{
    using Ptr = std::shared_ptr<Order>;

    Order() noexcept = default;

    Order(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        Currency currency = Currency::BASE) noexcept;

    [[nodiscard]] OrderDirection direction() const noexcept { return m_direction; }
    [[nodiscard]] STPFlag stpFlag() const noexcept { return m_stpFlag; }
    [[nodiscard]] SettleFlag settleFlag() const noexcept { return m_settleFlag; }
    [[nodiscard]] Currency currency() const noexcept { return m_currency; }

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    OrderDirection m_direction;
    STPFlag m_stpFlag{STPFlag::CO};
    SettleFlag m_settleFlag{SettleType::FIFO};
    Currency m_currency{Currency::BASE};

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency));
};

//-------------------------------------------------------------------------

struct MarketOrder : public Order
{
    using Ptr = std::shared_ptr<MarketOrder>;

    MarketOrder() noexcept = default;

    MarketOrder(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        Currency currency = Currency::BASE) noexcept;

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency));
};

//-------------------------------------------------------------------------

struct LimitOrder : public Order
{
    using Ptr = std::shared_ptr<LimitOrder>;

    LimitOrder() noexcept = default;

    LimitOrder(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t price,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = std::nullopt,
        Currency currency = Currency::BASE) noexcept;

    [[nodiscard]] taosim::decimal_t price() const noexcept { return m_price; };
    [[nodiscard]] bool postOnly() const noexcept { return m_postOnly; }
    [[nodiscard]] taosim::TimeInForce timeInForce() const noexcept { return m_timeInForce; }
    [[nodiscard]] std::optional<Timestamp> expiryPeriod() const noexcept { return m_expiryPeriod; }

    void setPrice(taosim::decimal_t newPrice);

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json, int priceDecimals, int volumeDecimals);

    taosim::decimal_t m_price;
    bool m_postOnly{};
    taosim::TimeInForce m_timeInForce{taosim::TimeInForce::GTC};
    std::optional<Timestamp> m_expiryPeriod{};

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency),
        MSGPACK_NVP("price", m_price),
        MSGPACK_NVP("postOnly", m_postOnly),
        MSGPACK_NVP("timeInForce", m_timeInForce),
        MSGPACK_NVP("expiryPeriod", m_expiryPeriod));
};

//-------------------------------------------------------------------------

struct OrderClientContext
{
    AgentId agentId;
    std::optional<ClientOrderID> clientOrderId;

    OrderClientContext() noexcept = default;

    OrderClientContext(AgentId agentId, std::optional<ClientOrderID> clientOrderId = {}) noexcept
        : agentId{agentId}, clientOrderId{clientOrderId}
    {}

    [[nodiscard]] static OrderClientContext fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentId, clientOrderId);
};

//-------------------------------------------------------------------------

struct OrderContext : public JsonSerializable
{
    AgentId agentId;
    BookId bookId;
    std::optional<ClientOrderID> clientOrderId;

    OrderContext() = default;

    OrderContext(
        AgentId agentId, BookId bookId, std::optional<ClientOrderID> clientOrderId = {}) noexcept
        : agentId{agentId}, bookId{bookId}, clientOrderId{clientOrderId}
    {}

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static OrderContext fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentId, bookId, clientOrderId);
};

//-------------------------------------------------------------------------

struct OrderLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<OrderLogContext>;

    AgentId agentId;
    BookId bookId;

    OrderLogContext() noexcept = default;

    OrderLogContext(AgentId agentId, BookId bookId) noexcept
        : agentId{agentId}, bookId{bookId}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(agentId, bookId);
};

//-------------------------------------------------------------------------

struct OrderWithLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<OrderWithLogContext>;

    Order::Ptr order;
    OrderLogContext::Ptr logContext;

    OrderWithLogContext(Order::Ptr order, OrderLogContext::Ptr logContext) noexcept
        : order{order}, logContext{logContext}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(order, logContext);
};

//-------------------------------------------------------------------------
