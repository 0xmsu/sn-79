/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/decimal.hpp>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

struct Fees
{
    decimal_t maker{};
    decimal_t taker{};

    Fees operator+(const Fees& other) const noexcept
    {
        return {
            .maker = maker + other.maker,
            .taker = taker + other.taker
        };
    }

    Fees operator-(const Fees& other) const noexcept
    {
        return {
            .maker = maker - other.maker,
            .taker = taker - other.taker
        };
    }

    Fees& operator+=(const Fees& other) noexcept
    {
        maker += other.maker;
        taker += other.taker;
        return *this;
    }

    Fees& operator-=(const Fees& other) noexcept
    {
        maker -= other.maker;
        taker -= other.taker;
        return *this;
    }

    MSGPACK_DEFINE_MAP(maker, taker);
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------