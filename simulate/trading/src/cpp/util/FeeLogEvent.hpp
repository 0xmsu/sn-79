/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.hpp"
#include <taosim/exchange/Fees.hpp>

//-------------------------------------------------------------------------

namespace taosim
{

struct FeeLogEvent
{
    BookId bookId;
    AgentId restingAgentId;
    AgentId aggressingAgentId;
    exchange::Fees fees;
    decimal_t price;
    decimal_t volume;
    decimal_t restingRatio;
    decimal_t aggressingRatio;
};

}  // namespace taosim

//-------------------------------------------------------------------------
