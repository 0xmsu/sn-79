/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/Loan.hpp>

#include <algorithm>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

Loan::Loan(const LoanDesc& desc) noexcept
    : m_amount{desc.amount},
      m_direction{desc.direction},
      m_leverage{desc.leverage},
      m_collateral{desc.collateral},
      m_marginCallPrice{desc.marginCallPrice}
{}

//-------------------------------------------------------------------------

Loan& Loan::operator+=(const Loan& other) noexcept
{
    m_amount += other.m_amount;
    m_collateral += other.m_collateral;
    return *this;
}

//-------------------------------------------------------------------------

Collateral Loan::settle(decimal_t amount, decimal_t price, const RoundParams& roundParams)
{
    amount = util::round(
        amount,
        m_direction == OrderDirection::BUY
            ? roundParams.quoteDecimals : roundParams.baseDecimals);

    if (amount == m_amount) {
        m_amount = {};
        return std::exchange(m_collateral, {});
    }
    if (amount > m_amount) [[unlikely]] {
        throw std::runtime_error{fmt::format(
            "{}: amount ({}) greater than m_amount ({})",
            std::source_location::current().function_name(), amount, m_amount)};
    }

    const decimal_t r = amount / m_amount;
    m_amount -= amount;

    const decimal_t q1 = m_collateral.base() * price / m_collateral.valueInQuote(price);
    const decimal_t q2 = util::dec1m(q1);

    if (m_direction == OrderDirection::BUY) {
        const decimal_t baseCollateralToRelease = r < q1
            ? util::round(r / q1 * m_collateral.base(), roundParams.baseDecimals)
            : m_collateral.base();
        m_collateral.base() -= baseCollateralToRelease;
        if (r <= q1) {
            return Collateral({.base = baseCollateralToRelease});
        }
        const decimal_t rPrime = r - q1;
        const decimal_t quoteCollateralToRelease =
            util::round(rPrime / q2 * m_collateral.quote(), roundParams.quoteDecimals);
        m_collateral.quote() -= quoteCollateralToRelease;
        return Collateral({
            .base = baseCollateralToRelease,
            .quote = quoteCollateralToRelease
        });
    }
    else {
        const decimal_t quoteCollateralToRelease = r < q2
            ? util::round(r / q2 * m_collateral.quote(), roundParams.quoteDecimals)
            : m_collateral.quote();
        m_collateral.quote() -= quoteCollateralToRelease;
        if (r <= q2) {
            return Collateral({.quote = quoteCollateralToRelease});
        }
        const decimal_t rPrime = r - q2;
        const decimal_t baseCollateralToRelease =
            util::round(rPrime / q1 * m_collateral.base(), roundParams.baseDecimals);
        m_collateral.base() -= baseCollateralToRelease;
        return Collateral({
            .base = baseCollateralToRelease,
            .quote = quoteCollateralToRelease
        });
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
