/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <bdldfp_decimal.h>
#include <bdldfp_decimalconvertutil.h>
#include <bdldfp_decimalutil.h>
#include <fmt/format.h>

#include <spanstream>

//-------------------------------------------------------------------------

#define DEC(lit) BDLDFP_DECIMAL_DD(lit)

//-------------------------------------------------------------------------

namespace taosim
{

using decimal_t = BloombergLP::bdldfp::Decimal128;

struct PackedDecimal
{
    uint8_t data[sizeof(decimal_t)]{};
};

}  // namespace taosim

//-------------------------------------------------------------------------

namespace taosim::util
{

inline constexpr uint32_t kDefaultDecimalPlaces = 8;

[[nodiscard]] inline decimal_t round(
    decimal_t val, uint32_t decimalPlaces = kDefaultDecimalPlaces)
{
    return BloombergLP::bdldfp::DecimalUtil::trunc(val, decimalPlaces);
}

[[nodiscard]] inline decimal_t roundUp(decimal_t val, uint32_t decimalPlaces)
{
    using namespace BloombergLP::bdldfp;
    const auto factor = DecimalUtil::multiplyByPowerOf10(decimal_t{1}, decimalPlaces);
    return DecimalUtil::ceil(val * factor) / factor;
}

[[nodiscard]] inline double decimal2double(decimal_t val)
{
    return BloombergLP::bdldfp::DecimalConvertUtil::decimalToDouble(val);
}

[[nodiscard]] inline decimal_t double2decimal(
    double val, uint32_t decimalPlaces = kDefaultDecimalPlaces)
{
    return round(decimal_t{val}, decimalPlaces);
}

[[nodiscard]] inline PackedDecimal packDecimal(decimal_t val)
{
    PackedDecimal packed;
    BloombergLP::bdldfp::DecimalConvertUtil::decimalToDPD(packed.data, val);
    return packed;
}

[[nodiscard]] inline decimal_t unpackDecimal(PackedDecimal val)
{
    decimal_t unpacked;
    BloombergLP::bdldfp::DecimalConvertUtil::decimalFromDPD(&unpacked, val.data);
    return unpacked;
}

[[nodiscard]] inline decimal_t fma(decimal_t a, decimal_t b, decimal_t c) noexcept
{
    return BloombergLP::bdldfp::DecimalUtil::fma(a, b, c);
}

[[nodiscard]] inline decimal_t pow(decimal_t a, decimal_t b)
{
    return BloombergLP::bdldfp::DecimalUtil::pow(a, b);
}

[[nodiscard]] inline decimal_t dec1p(decimal_t val) noexcept
{
    return 1 + val;
}

[[nodiscard]] inline decimal_t dec1m(decimal_t val) noexcept
{
    return 1 - val;
}

[[nodiscard]] inline decimal_t decInv1p(decimal_t val) noexcept
{
    return 1 / dec1p(val);
}

[[nodiscard]] inline decimal_t abs(decimal_t val) noexcept
{
    return val < decimal_t{} ? -val : val;
}

}  // namespace taosim::util

//-------------------------------------------------------------------------

namespace taosim::literals
{

[[nodiscard]] constexpr decimal_t operator"" _dec(unsigned long long int val)
{
    return decimal_t{val};
}

}  // namespace taosim::literals

//-------------------------------------------------------------------------

static inline void trim(std::span<char> span)
{
    const size_t len = std::strlen(span.data());
    if (len <= 3uz) return;
    size_t i = len - 1;
    while (i > 1 && span[i] == '0' && span[i - 1] != '.') {
        --i;
    }
    span[i + 1] = '\0';
}

template<>
struct fmt::formatter<taosim::decimal_t>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(taosim::decimal_t val, FormatContext& ctx) const
    {
        using namespace taosim::literals;
        char buf[64]{};
        std::ospanstream oss{buf};
        if (val == 0_dec) [[unlikely]] {
            oss << "0.0";
        } else {
            oss << val;
            trim(buf);
        }
        return fmt::format_to(ctx.out(), "{}", buf);
    }
};

//-------------------------------------------------------------------------
