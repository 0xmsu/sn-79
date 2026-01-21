/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/Collateral.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{
    
//-------------------------------------------------------------------------

Collateral& Collateral::operator+=(const Collateral& other) noexcept
{
    m.base += other.m.base;
    m.quote += other.m.quote;
    return *this;
}

//-------------------------------------------------------------------------

Collateral& Collateral::operator-=(const Collateral& other) noexcept
{
    m.base -= other.m.base;
    m.quote -= other.m.quote;
    return *this;
}

//-------------------------------------------------------------------------

bool Collateral::operator==(const Collateral& other) const noexcept
{
    return m.base == other.m.base && m.quote == other.m.quote;
}

//-------------------------------------------------------------------------

decimal_t Collateral::valueInBase(decimal_t price) const noexcept
{
    return m.base + m.quote / price;
}

//-------------------------------------------------------------------------

decimal_t Collateral::valueInQuote(decimal_t price) const noexcept
{
    return util::fma(m.base, price, m.quote);
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
