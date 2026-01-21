/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/decimal/serialization/decimal.hpp"
#include "taosim/serialization/msgpack/common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

//-------------------------------------------------------------------------

struct DecimalSerializationTest : TestWithParam<decimal_t>
{
    virtual void SetUp() override
    {
        refValue = GetParam();
    }

    decimal_t refValue;
};

//-------------------------------------------------------------------------

TEST_P(DecimalSerializationTest, Double)
{
    static const decimal_t epsilon = DEC(1e-7);
    serialization::HumanReadableStream stream;
    msgpack::pack(stream, refValue);
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();
    EXPECT_TRUE(util::abs(deserialized.as<decimal_t>() - refValue) < epsilon);
}

TEST_P(DecimalSerializationTest, Packed)
{
    serialization::BinaryStream stream;
    msgpack::pack(stream, refValue);
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();
    EXPECT_EQ(deserialized.as<decimal_t>(), refValue);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalSerializationTests,
    DecimalSerializationTest,
    Values(
        DEC(-293.497),
        DEC(-4.2e-18),
        DEC(3.22),
        DEC(13.37),
        DEC(6.8392581e8)
    ));

//-------------------------------------------------------------------------