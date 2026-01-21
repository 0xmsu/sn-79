/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include <taosim/exchange/FeePolicyWrapper.hpp>
#include <taosim/exchange/OrderPlacementValidator.hpp>

#include <map>
#include <memory>
#include <set>
#include <variant>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

struct MarketOrderDesc
{
    std::variant<AgentId, LocalAgentId> agentId;
    PlaceOrderMarketPayload::Ptr payload;
};

struct LimitOrderDesc
{
    std::variant<AgentId, LocalAgentId> agentId;
    PlaceOrderLimitPayload::Ptr payload;
};

using OrderDesc = std::variant<
    MarketOrderDesc,
    LimitOrderDesc
>;

struct CancelOrderDesc
{
    BookId bookId;
    LimitOrder::Ptr order;
    decimal_t volumeToCancel;
};

struct ClosePositionDesc
{
    BookId bookId;
    AgentId agentId;
    OrderID orderId;
    std::optional<decimal_t> volumeToClose;
};

struct OrderResult
{
    OrderErrorCode ec;
    decimal_t orderSize;
};

//-------------------------------------------------------------------------

class ClearingManager
{
public:
    struct MarginCallContext
    {
        OrderID orderId;
        AgentId agentId;
    };

    using MarginCallContainer = std::vector<std::map<decimal_t, std::vector<MarginCallContext>>>;

    explicit ClearingManager(
        MultiBookExchangeAgent* exchange,
        size_t bookCount,
        std::unique_ptr<FeePolicyWrapper> feePolicy,
        OrderPlacementValidator::Parameters validatorParams) noexcept;

    [[nodiscard]] MultiBookExchangeAgent* exchange() noexcept;
    [[nodiscard]] accounting::AccountRegistry& accounts() noexcept;

    [[nodiscard]] OrderResult handleOrder(const OrderDesc& orderDesc);
    void handleCancelOrder(const CancelOrderDesc& cancelDesc);
    bool handleClosePosition(const ClosePositionDesc& closeDesc);
    Fees handleTrade(const TradeDesc& tradeDesc);
    void updateFeeTiers(Timestamp time) noexcept;

    [[nodiscard]] auto&& marginBuys(this auto&& self) noexcept { return self.m_marginBuy; }
    [[nodiscard]] auto&& marginSells(this auto&& self) noexcept { return self.m_marginSell; }
    [[nodiscard]] FeePolicyWrapper* feePolicy() const noexcept { return m_feePolicy.get(); };

private:
    MultiBookExchangeAgent* m_exchange;
    std::unique_ptr<FeePolicyWrapper> m_feePolicy;
    MarginCallContainer m_marginBuy;
    MarginCallContainer m_marginSell;
    OrderPlacementValidator m_orderPlacementValidator;

    void removeMarginOrders(
        BookId bookId, OrderDirection direction, std::span<std::pair<OrderID, taosim::decimal_t>> ids);
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------