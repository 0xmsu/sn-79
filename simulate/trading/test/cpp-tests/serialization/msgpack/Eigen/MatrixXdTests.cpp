/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/serialization/msgpack/Eigen/MatrixXd.hpp>
#include <taosim_tests/serialization/msgpack/common.hpp>

#include <gmock/gmock.h>

//-------------------------------------------------------------------------

using namespace testing;

//-------------------------------------------------------------------------

struct MatrixXdRoundTripTest : public TestWithParam<Eigen::MatrixXd>
{
    virtual void SetUp() override
    {
        refValue = GetParam();
    }

    Eigen::MatrixXd refValue;
};

TEST_P(MatrixXdRoundTripTest, WorksCorrectly)
{
    const auto value = taosim_tests::serialization::msgpack::performRoundTrip(refValue);

    EXPECT_EQ(value.rows(), refValue.rows());
    EXPECT_EQ(value.cols(), refValue.cols());
    EXPECT_EQ(value, refValue);
}

INSTANTIATE_TEST_SUITE_P(
    MatrixXdTest,
    MatrixXdRoundTripTest,
    Values(
        Eigen::Matrix2d::Random(),
        Eigen::Matrix3d::Random(),
        Eigen::Matrix4d::Random(),
        Eigen::MatrixXd::Random(32, 32),
        Eigen::MatrixXd::Random(128, 64)
    ));

//-------------------------------------------------------------------------