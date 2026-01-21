/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/Book.hpp>
#include <taosim/decimal/serialization/decimal.hpp>
#include "Cancellation.hpp"
#include "ClosePosition.hpp"
#include <taosim/message/MessagePayload.hpp>
#include "Order.hpp"
#include "Trade.hpp"
#include "common.hpp"
#include "Flags.hpp"

#include <msgpack.hpp>

#include <optional>
#include <vector>

using STPFlag = taosim::STPFlag;
using SettleFlag = taosim::SettleFlag;
using SettleType = taosim::SettleType;

//-------------------------------------------------------------------------

struct StartSimulationPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<StartSimulationPayload>;

    std::string logDir;

    StartSimulationPayload() noexcept = default;

    StartSimulationPayload(const std::string& logDir) noexcept
        : logDir{logDir}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(logDir);
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketPayload>;

    OrderDirection direction;
    taosim::decimal_t volume;
    taosim::decimal_t leverage;
    BookId bookId;
    Currency currency;
    std::optional<ClientOrderID> clientOrderId{};
    STPFlag stpFlag{STPFlag::CO};
    SettleFlag settleFlag{SettleType::FIFO};

    PlaceOrderMarketPayload() noexcept = default;

    PlaceOrderMarketPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO) noexcept
        : direction{direction},
          volume{volume},
          leverage{0_dec},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          stpFlag{stpFlag},
          settleFlag{settleFlag}
    {}

    PlaceOrderMarketPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t leverage,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO) noexcept
        : direction{direction},
          volume{volume},
          leverage{leverage},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          stpFlag{stpFlag},
          settleFlag{settleFlag}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        direction,
        volume,
        leverage,
        bookId,
        currency,
        clientOrderId,
        stpFlag,
        settleFlag);
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketResponsePayload>;

    OrderID id;
    PlaceOrderMarketPayload::Ptr requestPayload;

    PlaceOrderMarketResponsePayload() noexcept = default;

    PlaceOrderMarketResponsePayload(
        OrderID id, PlaceOrderMarketPayload::Ptr requestPayload) noexcept
        : id{id}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id), requestPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketErrorResponsePayload>;

    PlaceOrderMarketPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    PlaceOrderMarketErrorResponsePayload() noexcept = default;

    PlaceOrderMarketErrorResponsePayload(
        PlaceOrderMarketPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : requestPayload{requestPayload}, errorPayload{std::move(errorPayload)}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitPayload>;

    OrderDirection direction;
    taosim::decimal_t volume;
    taosim::decimal_t price;
    taosim::decimal_t leverage{};
    BookId bookId;
    Currency currency;
    std::optional<ClientOrderID> clientOrderId{};
    bool postOnly{};
    taosim::TimeInForce timeInForce{taosim::TimeInForce::GTC};
    std::optional<Timestamp> expiryPeriod{};
    STPFlag stpFlag{STPFlag::CO};
    SettleFlag settleFlag{SettleType::FIFO};

    PlaceOrderLimitPayload() noexcept = default;

    PlaceOrderLimitPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t price,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = std::nullopt,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO) noexcept
        : direction{direction},
          volume{volume},
          price{price},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          postOnly{postOnly},
          timeInForce{timeInForce},
          expiryPeriod{expiryPeriod},
          stpFlag{stpFlag},
          settleFlag{settleFlag}
    {}

    PlaceOrderLimitPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t price,
        taosim::decimal_t leverage,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = std::nullopt,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO) noexcept
        : direction{direction},
          volume{volume},
          price{price},
          leverage{leverage},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          postOnly{postOnly},
          timeInForce{timeInForce},
          expiryPeriod{expiryPeriod},
          stpFlag{stpFlag},
          settleFlag{settleFlag}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        direction,
        volume,
        price,
        leverage,
        bookId,
        currency,
        clientOrderId,
        postOnly,
        timeInForce,
        expiryPeriod,
        stpFlag,
        settleFlag);
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitResponsePayload>;

    OrderID id;
    PlaceOrderLimitPayload::Ptr requestPayload;

    PlaceOrderLimitResponsePayload() noexcept = default;

    PlaceOrderLimitResponsePayload(OrderID id, PlaceOrderLimitPayload::Ptr requestPayload) noexcept
        : id{id}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id), requestPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitErrorResponsePayload>;

    PlaceOrderLimitPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    PlaceOrderLimitErrorResponsePayload() noexcept = default;

    PlaceOrderLimitErrorResponsePayload(
        PlaceOrderLimitPayload::Ptr requestPayload, ErrorResponsePayload::Ptr errorPayload) noexcept
        : requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct RetrieveOrdersPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveOrdersPayload>;

    std::vector<OrderID> ids;
    BookId bookId;

    RetrieveOrdersPayload() noexcept = default;

    RetrieveOrdersPayload(std::vector<OrderID> ids, BookId bookId) noexcept
        : ids{std::move(ids)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderIds", ids), bookId);
};

//-------------------------------------------------------------------------

struct RetrieveOrdersResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveOrdersResponsePayload>;

    std::vector<LimitOrder> orders;
    BookId bookId;

    RetrieveOrdersResponsePayload() noexcept = default;

    RetrieveOrdersResponsePayload(std::vector<LimitOrder> orders, BookId bookId) noexcept
        : orders{std::move(orders)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orders, bookId);
};

//-------------------------------------------------------------------------

struct CancelOrdersPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersPayload>;

    std::vector<taosim::event::Cancellation> cancellations;
    BookId bookId;

    CancelOrdersPayload() noexcept = default;

    CancelOrdersPayload(
        std::vector<taosim::event::Cancellation> cancellations, BookId bookId) noexcept
        : cancellations{std::move(cancellations)}, bookId{bookId}
    {}

    CancelOrdersPayload(taosim::event::Cancellation cancellation, BookId bookId) noexcept
        : cancellations{cancellation}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(cancellations, bookId);
};

//-------------------------------------------------------------------------

struct CancelOrdersResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersResponsePayload>;

    std::vector<OrderID> orderIds;
    CancelOrdersPayload::Ptr requestPayload;

    CancelOrdersResponsePayload() noexcept = default;

    CancelOrdersResponsePayload(
        std::vector<OrderID> orderIds,
        CancelOrdersPayload::Ptr requestPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload);
};

//-------------------------------------------------------------------------

struct CancelOrdersErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersErrorResponsePayload>;

    std::vector<OrderID> orderIds;
    CancelOrdersPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    CancelOrdersErrorResponsePayload() noexcept = default;

    CancelOrdersErrorResponsePayload(
        std::vector<OrderID> orderIds,
        CancelOrdersPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct ClosePositionsPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsPayload>;

    std::vector<ClosePosition> closePositions;
    BookId bookId;

    ClosePositionsPayload() = default;

    ClosePositionsPayload(std::vector<ClosePosition> closePositions, BookId bookId) noexcept
        : closePositions{std::move(closePositions)}, bookId{bookId}
    {}

    ClosePositionsPayload(ClosePosition closePositions, BookId bookId) noexcept
        : closePositions{closePositions}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("closes", closePositions), bookId);
};

//-------------------------------------------------------------------------

struct ClosePositionsResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsResponsePayload>;

    std::vector<OrderID> orderIds;
    ClosePositionsPayload::Ptr requestPayload;

    ClosePositionsResponsePayload() noexcept = default;

    ClosePositionsResponsePayload(
        std::vector<OrderID> orderIds,
        ClosePositionsPayload::Ptr requestPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload);
};

//-------------------------------------------------------------------------

struct ClosePositionsErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsErrorResponsePayload>;

    std::vector<OrderID> orderIds;
    ClosePositionsPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    ClosePositionsErrorResponsePayload() noexcept = default;

    ClosePositionsErrorResponsePayload(
        std::vector<OrderID> orderIds,
        ClosePositionsPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct RetrieveL2Payload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL2Payload>;

    size_t depth;
    BookId bookId;

    RetrieveL2Payload() noexcept = default;

    RetrieveL2Payload(size_t depth, BookId bookId) noexcept
        : depth{depth}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(depth, bookId);
};

//-------------------------------------------------------------------------

struct BookLevel
{
    taosim::decimal_t price;
    taosim::decimal_t quantity;

    MSGPACK_DEFINE_MAP(price, quantity);
};

struct RetrieveL2ResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL2ResponsePayload>;

    Timestamp time;
    std::vector<BookLevel> bids;
    std::vector<BookLevel> asks;
    BookId bookId;

    RetrieveL2ResponsePayload() noexcept = default;

    RetrieveL2ResponsePayload(
        Timestamp time,
        std::vector<BookLevel> bids,
        std::vector<BookLevel> asks,
        BookId bookId) noexcept
        : time{time}, bids{std::move(bids)}, asks{std::move(asks)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(time, bids, asks, bookId);
};

//-------------------------------------------------------------------------

struct RetrieveL1Payload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1Payload>;

    BookId bookId;

    RetrieveL1Payload() = default;

    RetrieveL1Payload(BookId bookId) noexcept : bookId{bookId} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId);
};

//-------------------------------------------------------------------------

struct RetrieveL1ResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1ResponsePayload>;

    Timestamp time{};
    taosim::decimal_t bestAskPrice{};
    taosim::decimal_t bestAskVolume{};
    taosim::decimal_t askTotalVolume{};
    taosim::decimal_t bestBidPrice{};
    taosim::decimal_t bestBidVolume{};
    taosim::decimal_t bidTotalVolume{};
    BookId bookId;

    RetrieveL1ResponsePayload() noexcept = default;

    RetrieveL1ResponsePayload(Timestamp time, BookId bookId) noexcept
        : time{time}, bookId{bookId}
    {}

    RetrieveL1ResponsePayload(
        Timestamp time,
        taosim::decimal_t bestAskPrice,
        taosim::decimal_t bestAskVolume,
        taosim::decimal_t askTotalVolume,
        taosim::decimal_t bestBidPrice,
        taosim::decimal_t bestBidVolume,
        taosim::decimal_t bidTotalVolume,
        BookId bookId) noexcept
        : time{time},
          bestAskPrice{bestAskPrice},
          bestAskVolume{bestAskVolume},
          askTotalVolume{askTotalVolume},
          bestBidPrice{bestBidPrice},
          bestBidVolume{bestBidVolume},
          bidTotalVolume{bidTotalVolume},
          bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("timestamp", time),
        bestAskPrice,
        bestAskVolume,
        askTotalVolume,
        bestBidPrice,
        bestBidVolume,
        bidTotalVolume,
        bookId);
};

//-------------------------------------------------------------------------

struct SubscribeEventTradeByOrderPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<SubscribeEventTradeByOrderPayload>;

    OrderID id;

    SubscribeEventTradeByOrderPayload() noexcept = default;

    SubscribeEventTradeByOrderPayload(OrderID id) noexcept : id{id} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id));
};

//-------------------------------------------------------------------------

struct EventOrderMarketPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventOrderMarketPayload>;

    MarketOrder order;

    EventOrderMarketPayload() noexcept = default;

    EventOrderMarketPayload(const MarketOrder& order) noexcept : order{order} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(order);
};

//-------------------------------------------------------------------------

struct EventOrderLimitPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventOrderLimitPayload>;

    LimitOrder order;

    EventOrderLimitPayload() noexcept = default;

    EventOrderLimitPayload(const LimitOrder& order) noexcept : order{order} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(order);
};

//-------------------------------------------------------------------------

struct EventTradePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventTradePayload>;

    Trade trade;
    TradeLogContext context;
    BookId bookId;
    std::optional<ClientOrderID> clientOrderId{};

    EventTradePayload() noexcept = default;

    EventTradePayload(
        const Trade& trade,
        const TradeLogContext& context,
        BookId bookId,
        std::optional<ClientOrderID> clientOrderId = {}) noexcept
        : trade{trade}, context{context}, bookId{bookId}, clientOrderId{clientOrderId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(trade, context, bookId, clientOrderId);
};

//-------------------------------------------------------------------------

struct ResetAgentsPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsPayload>;

    std::vector<AgentId> agentIds;

    ResetAgentsPayload() noexcept = default;

    ResetAgentsPayload(std::vector<AgentId> agentIds) noexcept
        : agentIds{std::move(agentIds)}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds);
};

//-------------------------------------------------------------------------

struct ResetAgentsResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsResponsePayload>;

    std::vector<AgentId> agentIds;
    ResetAgentsPayload::Ptr requestPayload;

    ResetAgentsResponsePayload() noexcept = default;

    ResetAgentsResponsePayload(
        std::vector<AgentId> agentIds,ResetAgentsPayload::Ptr requestPayload) noexcept
        : agentIds{std::move(agentIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds, requestPayload);
};

//-------------------------------------------------------------------------

struct ResetAgentsErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsErrorResponsePayload>;

    std::vector<AgentId> agentIds;
    ResetAgentsPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    ResetAgentsErrorResponsePayload() noexcept = default;

    ResetAgentsErrorResponsePayload(
        std::vector<AgentId> agentIds,
        ResetAgentsPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : agentIds{std::move(agentIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------
