/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Order.hpp"
#include "Trade.hpp"
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::book 
{

class Book;

struct BookSignals
{
    UnsyncSignal<void(Order::Ptr, OrderContext)> orderCreated;
    UnsyncSignal<void(Order::Ptr, OrderContext)> orderLog;
    UnsyncSignal<void(LimitOrder::Ptr, OrderContext)> limitOrderProcessed;
    UnsyncSignal<void(MarketOrder::Ptr, OrderContext)> marketOrderProcessed;
    UnsyncSignal<void(Trade::Ptr, BookId)> trade;
    UnsyncSignal<void(OrderID, taosim::decimal_t)> cancel;
    UnsyncSignal<void(LimitOrder::Ptr, taosim::decimal_t, BookId)> cancelOrderDetails;
    UnsyncSignal<void(LimitOrder::Ptr, BookId)> unregister;
    UnsyncSignal<void(const taosim::book::Book*)> L2;
};

}  // namespace taosim::book

//-------------------------------------------------------------------------