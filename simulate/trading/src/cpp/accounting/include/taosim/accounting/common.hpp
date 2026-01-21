/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstdint>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

struct RoundParams
{
    uint32_t baseDecimals;
    uint32_t quoteDecimals;
};

// TODO: Pass stacktrace instead.
uint32_t validateDecimalPlaces(
    uint32_t decimalPlaces, std::source_location sl = std::source_location::current());

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
