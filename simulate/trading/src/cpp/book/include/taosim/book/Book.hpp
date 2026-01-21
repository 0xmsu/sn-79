/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "CSVPrintable.hpp"
#include "IHumanPrintable.hpp"

#include <taosim/book/BookSignals.hpp>
#include <taosim/book/OrderContainer.hpp>
#include <taosim/book/TickContainer.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class Book : public CSVPrintable, public JsonSerializable
{
public:
    using Ptr = std::shared_ptr<Book>;

    Book(Simulation* simulation, BookId id, size_t maxDepth, size_t detailedDepth);

    [[nodiscard]] BookId id() const noexcept { return m_id; }
    [[nodiscard]] size_t maxDepth() const noexcept { return m_maxDepth; }
    [[nodiscard]] size_t detailedDepth() const noexcept { return m_detailedDepth; }
    [[nodiscard]] BookSignals& signals() noexcept { return m_signals; }
    [[nodiscard]] auto&& buyQueue(this auto&& self) noexcept { return self.m_buyQueue; }
    [[nodiscard]] auto&& sellQueue(this auto&& self) noexcept { return self.m_sellQueue; }
    [[nodiscard]] auto&& orderIdCounter(this auto&& self) noexcept { return self.m_orderIdCounter; }
    [[nodiscard]] auto&& tradeIdCounter(this auto&& self) noexcept { return self.m_tradeIdCounter; }
    [[nodiscard]] auto&& orderToClientInfo(this auto&& self) noexcept { return self.m_order2clientCtx; }
    [[nodiscard]] auto&& orderIdMap(this auto&& self) noexcept { return self.m_orderIdMap; }
    [[nodiscard]] taosim::decimal_t midPrice() const noexcept;
    [[nodiscard]] taosim::decimal_t bestBid() const noexcept;
    [[nodiscard]] taosim::decimal_t bestAsk() const noexcept;

    template<typename... Args>
    [[nodiscard]] MarketOrder::Ptr placeMarketOrder(OrderClientContext ctx, Args&&... args);

    template<typename... Args>
    [[nodiscard]] LimitOrder::Ptr placeLimitOrder(OrderClientContext ctx, Args&&... args);

    template<typename... Args>
    void logTrade(Args&&... args);

    void placeOrder(MarketOrder::Ptr order);
    void placeOrder(LimitOrder::Ptr order);
    void placeLimitBuy(LimitOrder::Ptr order);
    void placeLimitSell(LimitOrder::Ptr order);
    bool cancelOrder(OrderID orderId, std::optional<taosim::decimal_t> volumeToCancel = {});
    [[nodiscard]] std::optional<LimitOrder::Ptr> getOrder(OrderID orderId) const;
    void registerLimitOrder(LimitOrder::Ptr order);
    void unregisterLimitOrder(LimitOrder::Ptr order);
    taosim::decimal_t processAgainstTheBuyQueue(Order::Ptr order, taosim::decimal_t minPrice);
    taosim::decimal_t processAgainstTheSellQueue(Order::Ptr order, taosim::decimal_t maxPrice);
    [[nodiscard]] taosim::book::TickContainer* preventSelfTrade(
        taosim::book::TickContainer* queue, LimitOrder::Ptr iop, Order::Ptr order, AgentId agentId);
    void printCSV(uint32_t depth) const;

    virtual void printCSV() const override { printCSV(5); }
    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

private:
    void dumpCSVLOB(auto begin, auto end, uint32_t depth) const;

    void emitL2Signal([[maybe_unused]] auto&&... args) const { m_signals.L2(this); }
    void setupL2Signal();

    Simulation* m_simulation;
    BookId m_id;
    size_t m_maxDepth;
    size_t m_detailedDepth;
    BookSignals m_signals;
    OrderID m_orderIdCounter{};
    TradeID m_tradeIdCounter{};
    std::map<OrderID, LimitOrder::Ptr> m_orderIdMap;
    std::map<OrderID, OrderClientContext> m_order2clientCtx;
    taosim::book::OrderContainer m_buyQueue;
    taosim::book::OrderContainer m_sellQueue;
    bool m_initMode = false;

    friend class ::Simulation;
};

//-------------------------------------------------------------------------

template<typename... Args>
MarketOrder::Ptr Book::placeMarketOrder(OrderClientContext clientCtx, Args&&... args)
{
    static_assert(std::constructible_from<MarketOrder, OrderID, Args...>);
    const auto marketOrder =
        std::make_shared<MarketOrder>(m_orderIdCounter++, std::forward<Args>(args)...);
    m_order2clientCtx.insert({marketOrder->id(), clientCtx});
    m_signals.orderCreated(
        marketOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    placeOrder(marketOrder);
    m_order2clientCtx.erase(marketOrder->id());
    m_signals.orderLog(
        marketOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    return marketOrder;
}

//-------------------------------------------------------------------------

template<typename... Args>
LimitOrder::Ptr Book::placeLimitOrder(OrderClientContext clientCtx, Args&&... args)
{
    static_assert(std::constructible_from<LimitOrder, OrderID, Args...>);
    const auto limitOrder =
        std::make_shared<LimitOrder>(m_orderIdCounter++, std::forward<Args>(args)...);
    m_order2clientCtx.insert({limitOrder->id(), clientCtx});
    m_signals.orderCreated(
        limitOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    placeOrder(limitOrder);
    m_signals.orderLog(
        limitOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    return limitOrder;
}

//-------------------------------------------------------------------------

template<typename... Args>
void Book::logTrade(Args&&... args)
{
    static_assert(std::constructible_from<Trade, TradeID, Args...>);
    const auto trade = Trade::create(m_tradeIdCounter++, std::forward<Args>(args)...);
    m_signals.trade(trade, m_id);
}

//-------------------------------------------------------------------------

void Book::dumpCSVLOB(auto begin, auto end, uint32_t depth) const
{
    while (depth > 0 && begin != end) {
        const taosim::decimal_t totalVolume = [&] {
            taosim::decimal_t totalVolume{};
            for (auto it = begin->begin(); it != begin->end(); ++it) {
                LimitOrder::Ptr order = *it;
                totalVolume += order->volume() * (1_dec + order->leverage());
            }
            return totalVolume;
        }();

        
        if (totalVolume > 0_dec) {
            std::cout
                << ","
                << begin->price()
                << ","
                << totalVolume;
        }

        --depth;
        ++begin;
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
