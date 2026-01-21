/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/ClearingManager.hpp>
#include "Simulation.hpp"

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

class Exchange
{
public:
    explicit Exchange(Simulation* simulation) noexcept;
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
