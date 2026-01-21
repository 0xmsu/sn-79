/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "FeeLogEvent.hpp"
#include "L3LogEvent.hpp"

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

class FeePolicyWrapper;

//-------------------------------------------------------------------------

struct ExchangeSignals
{
    UnsyncSignal<void(InstructionLogContext)> instructionLog;
    UnsyncSignal<void(OrderWithLogContext)> orderLog;
    UnsyncSignal<void(TradeWithLogContext)> tradeLog;
    UnsyncSignal<void(CancellationWithLogContext)> cancelLog;
    UnsyncSignal<void(taosim::L3LogEvent)> L3;
    UnsyncSignal<void(const FeePolicyWrapper*, taosim::FeeLogEvent)> feeLog;
    uint32_t eventCounter{};

    ExchangeSignals() noexcept;
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------