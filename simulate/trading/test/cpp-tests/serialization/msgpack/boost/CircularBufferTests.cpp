/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/serialization/msgpack/boost/circular_buffer.hpp>
#include <taosim_tests/serialization/msgpack/common.hpp>

#include <gmock/gmock.h>
#include <range/v3/view/iota.hpp>
#include <range/v3/to_container.hpp>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

namespace views = ranges::views;

//-------------------------------------------------------------------------

struct CircularBufferTestParams
{
    std::vector<size_t> data;
};

void PrintTo(const CircularBufferTestParams& params, std::ostream* os)
{
    internal::PrintTo(params.data, os);
}

struct CircularBufferRoundTripTest : TestWithParam<CircularBufferTestParams>
{
    virtual void SetUp() override
    {
        const auto& param = GetParam();
        refValue.assign(param.data.size(), param.data.cbegin(), param.data.cend());
    }

    boost::circular_buffer<size_t> refValue;
};

TEST_P(CircularBufferRoundTripTest, WorksCorrectly)
{
    const auto value = taosim_tests::serialization::msgpack::performRoundTrip(refValue);

    EXPECT_THAT(std::vector(value.begin(), value.end()), ContainerEq(GetParam().data));
}

INSTANTIATE_TEST_SUITE_P(
    CircularBufferTest,
    CircularBufferRoundTripTest,
    Values(
        CircularBufferTestParams{},
        CircularBufferTestParams{.data = {1, 2, 3}},
        CircularBufferTestParams{.data = views::iota(0uz, 20uz) | ranges::to<std::vector>}
    ));

//-------------------------------------------------------------------------