/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/serialization/msgpack/Eigen/VectorXd.hpp>
#include <taosim_tests/serialization/msgpack/common.hpp>

#include <gmock/gmock.h>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

//-------------------------------------------------------------------------

struct VectorXdRoundTripTest : public TestWithParam<Eigen::VectorXd>
{
    virtual void SetUp() override
    {
        refValue = GetParam();
    }

    Eigen::VectorXd refValue;
};

TEST_P(VectorXdRoundTripTest, WorksCorrectly)
{
    const auto value = taosim_tests::serialization::msgpack::performRoundTrip(refValue);

    EXPECT_EQ(value.size(), refValue.size());
    EXPECT_EQ(value, refValue);
}

INSTANTIATE_TEST_SUITE_P(
    VectorXdTest,
    VectorXdRoundTripTest,
    Values(
        Eigen::Vector2d::Random(),
        Eigen::Vector3d::Random(),
        Eigen::Vector4d::Random(),
        Eigen::VectorXd::Random(32),
        Eigen::VectorXd::Random(128)
    ));

//-------------------------------------------------------------------------