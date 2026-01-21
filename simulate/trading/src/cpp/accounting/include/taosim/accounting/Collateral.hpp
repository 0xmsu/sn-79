/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/decimal.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

struct CollateralDesc
{
    decimal_t base{};
    decimal_t quote{};
};

//-------------------------------------------------------------------------

class Collateral
{
public:
    Collateral() noexcept = default;
    explicit Collateral(const CollateralDesc& desc) noexcept : m{desc} {}

    Collateral& operator+=(const Collateral& other) noexcept;
    Collateral& operator-=(const Collateral& other) noexcept;

    [[nodiscard]] bool operator==(const Collateral& other) const noexcept;

    [[nodiscard]] auto&& base(this auto&& self) noexcept { return self.m.base; }
    [[nodiscard]] auto&& quote(this auto&& self) noexcept { return self.m.quote; }

    [[nodiscard]] decimal_t valueInBase(decimal_t price) const noexcept;
    [[nodiscard]] decimal_t valueInQuote(decimal_t price) const noexcept;

private:
    CollateralDesc m;
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::accounting::Collateral>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::accounting::Collateral& coll, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(), "Collateral{{.base = {}, .quote = {}}}", coll.base(), coll.quote());
    }
};

//-------------------------------------------------------------------------
