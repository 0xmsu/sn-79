/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/common.hpp>

#include <fmt/format.h>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

uint32_t validateDecimalPlaces(uint32_t decimalPlaces, std::source_location sl)
{
    if (!(decimalPlaces > 0)) {
        throw std::invalid_argument{fmt::format(
            "{}: decimalPlaces should be > 0, was {}", sl.function_name(), decimalPlaces)};
    }
    return decimalPlaces;
}

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
