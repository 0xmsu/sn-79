/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Timestamp.hpp>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

struct DelayBounds
{
    Timestamp min, max;
};

struct TimestampedPrice
{
    Timestamp timestamp{};
    double price{};
};

struct TopLevel
{
    double bid, ask;
};

struct TopLevelWithVolumes : TopLevel
{
    double bidQty, askQty;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------