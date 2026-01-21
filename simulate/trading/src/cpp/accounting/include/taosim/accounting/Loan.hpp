/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Collateral.hpp"
#include "Order.hpp"
#include <taosim/accounting/common.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

struct LoanDesc
{
    decimal_t amount{};
    OrderDirection direction{};
    decimal_t leverage{};
    Collateral collateral;
    decimal_t marginCallPrice{};
};

//-------------------------------------------------------------------------

class Loan
{
public:
    Loan() noexcept = default;
    explicit Loan(const LoanDesc& desc) noexcept;

    Loan& operator+=(const Loan& other) noexcept;

    [[nodiscard]] decimal_t amount() const noexcept { return m_amount; }
    [[nodiscard]] OrderDirection direction() const noexcept { return m_direction; }
    [[nodiscard]] decimal_t leverage() const noexcept { return m_leverage; }
    [[nodiscard]] const Collateral& collateral() const noexcept { return m_collateral; }
    [[nodiscard]] decimal_t marginCallPrice() const noexcept { return m_marginCallPrice; }

    Collateral settle(decimal_t amount, decimal_t price, const RoundParams& roundParams);

private:
    decimal_t m_amount;
    OrderDirection m_direction;
    decimal_t m_leverage;
    Collateral m_collateral;
    decimal_t m_marginCallPrice;
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
