/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/accounting/Loan.hpp"
#include "taosim/accounting/margin_utils.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------

static constexpr RoundParams s_roundParams{
    .baseDecimals = 4,
    .quoteDecimals = 8
};

struct CollateralCreationDesc
{
    decimal_t quoteValue;
    decimal_t price;
    decimal_t quoteProportion;
};

[[nodiscard]] static Collateral makeCollateral(const CollateralCreationDesc& desc)
{
    if (!(0_dec <= desc.quoteProportion && desc.quoteProportion <= 1_dec)) {
        throw std::invalid_argument{fmt::format(
            "{}: desc.quoteProportion should be in [0,1], was {}",
            std::source_location::current().function_name(), desc.quoteProportion)};
    }
    return Collateral({
        .base = util::round(
            util::dec1m(desc.quoteProportion) * desc.quoteValue / desc.price,
            s_roundParams.baseDecimals),
        .quote = util::round(desc.quoteProportion * desc.quoteValue, s_roundParams.quoteDecimals)
    });
}

namespace taosim::accounting
{

void PrintTo(const Collateral& coll, std::ostream* os)
{
    *os << fmt::format("{}", coll);
}

}  // namespace taosim::accounting

struct LoanCreationDesc
{
    decimal_t amount;
    OrderDirection direction;
    decimal_t leverage;
    decimal_t price;
    decimal_t collateralQuoteProportion;
    decimal_t maintenanceMargin;
};

[[nodiscard]] static Loan makeLoan(const LoanCreationDesc& desc)
{
    return Loan({
        .amount = desc.amount,
        .direction = desc.direction,
        .leverage = desc.leverage,
        .collateral = makeCollateral({
            .quoteValue = desc.amount / util::dec1p(desc.leverage)
                * (desc.direction == OrderDirection::BUY ? 1_dec : desc.price),
            .price = desc.price,
            .quoteProportion = desc.collateralQuoteProportion
        }),
        .marginCallPrice = calculateMarginCallPrice(
            desc.price, desc.leverage, desc.direction, desc.maintenanceMargin) 
    });
}

struct LoanTestParams
{
    LoanCreationDesc loanCreationDesc;
    decimal_t settleAmount;
    decimal_t price;
    Collateral refCollateral;
};

void PrintTo(const LoanTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.loanCreationDesc = {{.amount = {}, .direction = {}, .leverage = {},"
        ".price = {}, .collateralQuoteProportion = {}, .maintenanceMargin = {}}},"
        ".settleAmount = {}, .price = {}, .refCollateral = {}}}",
        params.loanCreationDesc.amount,
        params.loanCreationDesc.direction,
        params.loanCreationDesc.leverage,
        params.loanCreationDesc.price,
        params.loanCreationDesc.collateralQuoteProportion,
        params.loanCreationDesc.maintenanceMargin,
        params.settleAmount,
        params.price,
        params.refCollateral);
}

//-------------------------------------------------------------------------

TEST(LoanTests, SettleThrowsCorrectly)
{
    Loan loan = makeLoan({
        .amount = 10_dec,
        .direction = OrderDirection::BUY,
        .leverage = DEC(0.1),
        .price = DEC(0.5),
        .collateralQuoteProportion = DEC(0.75),
        .maintenanceMargin = DEC(0.25)
    });
    EXPECT_THROW(loan.settle(DEC(10.0001), 1_dec, s_roundParams), std::runtime_error);
}

//-------------------------------------------------------------------------

struct SettleTest : TestWithParam<LoanTestParams>
{
    virtual void SetUp() override
    {
        const auto& [loanCreationDesc, settleAmount, price, refCollateral] = GetParam();
        loan = makeLoan(loanCreationDesc);
        this->settleAmount = settleAmount;
        this->price = price;
        this->refCollateral = refCollateral;
    }

    Loan loan;
    decimal_t settleAmount;
    decimal_t price;
    Collateral refCollateral;
};

TEST_P(SettleTest, WorksCorrectly)
{
    const auto releasedCollateral = loan.settle(settleAmount, price, s_roundParams);
    EXPECT_EQ(releasedCollateral, refCollateral);
}

INSTANTIATE_TEST_SUITE_P(
    LoanTests,
    SettleTest,
    Values(
        LoanTestParams{
            .loanCreationDesc = {
                .amount = 1_dec,
                .direction = OrderDirection::BUY,
                .leverage = DEC(0.2),
                .price = DEC(0.3),
                .collateralQuoteProportion = 1_dec,
                .maintenanceMargin = DEC(0.25)
            },
            .settleAmount = DEC(0.25),
            .price = DEC(0.5),
            .refCollateral = Collateral({
                .base = 0_dec,
                .quote = DEC(0.20833333)
            })
        },
        LoanTestParams{
            .loanCreationDesc = {
                .amount = 5_dec,
                .direction = OrderDirection::SELL,
                .leverage = DEC(0.4),
                .price = DEC(0.3),
                .collateralQuoteProportion = 1_dec,
                .maintenanceMargin = DEC(0.25)
            },
            .settleAmount = DEC(2.35),
            .price = DEC(0.2),
            .refCollateral = Collateral({
                .base = 0_dec,
                .quote = DEC(0.50357142)
            })
        },
        LoanTestParams{
            .loanCreationDesc = {
                .amount = 5_dec,
                .direction = OrderDirection::BUY,
                .leverage = DEC(0.4),
                .price = DEC(0.3),
                .collateralQuoteProportion = 0_dec,
                .maintenanceMargin = DEC(0.25)
            },
            .settleAmount = 5_dec,
            .price = DEC(0.2),
            .refCollateral = makeCollateral({
                .quoteValue = 5_dec / util::dec1p(DEC(0.4)),
                .price = DEC(0.3),
                .quoteProportion = 0_dec
            })
        },
        LoanTestParams{
            .loanCreationDesc = {
                .amount = 10_dec,
                .direction = OrderDirection::BUY,
                .leverage = DEC(0.5),
                .price = DEC(2.5),
                .collateralQuoteProportion = DEC(0.75),
                .maintenanceMargin = DEC(0.25)
            },
            .settleAmount = DEC(3.33),
            .price = DEC(3.5),
            .refCollateral = Collateral({
                .base = DEC(0.6666),
                .quote = DEC(0.10882230)
            })
        },
        LoanTestParams{
            .loanCreationDesc = {
                .amount = 100_dec,
                .direction = OrderDirection::SELL,
                .leverage = DEC(0.75),
                .price = DEC(5.3745),
                .collateralQuoteProportion = DEC(0.42),
                .maintenanceMargin = DEC(0.25)
            },
            .settleAmount = DEC(57.89),
            .price = DEC(2.3498),
            .refCollateral = Collateral({
                .base = DEC(0.0),
                .quote = DEC(119.75527818)
            })
        }));

//-------------------------------------------------------------------------
