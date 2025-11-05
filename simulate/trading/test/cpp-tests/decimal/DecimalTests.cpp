/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/decimal/decimal.hpp"

#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

//-------------------------------------------------------------------------

struct RoundUpTestParams
{
    decimal_t value;
    uint32_t decimalPlaces;
    decimal_t refValue;
};

void PrintTo(const RoundUpTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.value = {}, .decimalPlaces = {}, .refValue = {}}}",
        params.value,
        params.decimalPlaces,
        params.refValue);
}

struct RoundUpTest : TestWithParam<RoundUpTestParams> {};

TEST_P(RoundUpTest, WorksCorrectly)
{
    const auto [value, decimalPlaces, refValue] = GetParam();
    EXPECT_EQ(util::roundUp(value, decimalPlaces), refValue);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalTests,
    RoundUpTest,
    Values(
        RoundUpTestParams{
            .value = DEC(42.32125839), .decimalPlaces = 3, .refValue = DEC(42.322)
        },
        RoundUpTestParams{
            .value = DEC(0.00005100), .decimalPlaces = 4, .refValue = DEC(0.0001)
        },
        RoundUpTestParams{
            .value = DEC(420.6921), .decimalPlaces = 2, .refValue = DEC(420.70)
        },
        RoundUpTestParams{
            .value = DEC(0.0), .decimalPlaces = 10, .refValue = DEC(0.0)
        },
        RoundUpTestParams{
            .value = DEC(-29358.2416619814), .decimalPlaces = 7, .refValue = DEC(-29358.2416619)
        },
        RoundUpTestParams{
            .value = DEC(-420.6921), .decimalPlaces = 2, .refValue = DEC(-420.69)
        },
        RoundUpTestParams{
            .value = DEC(10000.1), .decimalPlaces = 0, .refValue = DEC(10001.0)
        }
    ));

//-------------------------------------------------------------------------

struct PackUnpackTest : TestWithParam<decimal_t> {};

TEST_P(PackUnpackTest, WorksCorrectly)
{
    const decimal_t packee = GetParam();
    const auto packed = util::packDecimal(packee);
    const decimal_t unpacked = util::unpackDecimal(packed);
    EXPECT_EQ(packee, unpacked);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalTests,
    PackUnpackTest,
    Values(
        DEC(0.0),
        DEC(1.337),
        DEC(-32.2),
        DEC(42.0),
        DEC(-69420.0),
        DEC(1.234567890123456e-42)));

//-------------------------------------------------------------------------
