/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/util/SubscriptionRegistry.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <string>

//-------------------------------------------------------------------------

using namespace testing;

//-------------------------------------------------------------------------

TEST(SubscriptionRegistryTest, UInt)
{
    taosim::util::SubscriptionRegistry<uint32_t> reg;

    EXPECT_THAT(reg.add(0), IsTrue());
    EXPECT_THAT(reg.add(0), IsFalse());
    EXPECT_THAT(reg.add(42), IsTrue());
    EXPECT_THAT(reg.add(1337), IsTrue());
    EXPECT_THAT(reg.add(42), IsFalse());

    EXPECT_THAT(reg.subs(), ElementsAre(0, 42, 1337));
}

//-------------------------------------------------------------------------

TEST(SubscriptionRegistryTest, String)
{
    taosim::util::SubscriptionRegistry<std::string> reg;

    EXPECT_THAT(reg.add("foo"), IsTrue());
    EXPECT_THAT(reg.add("foo"), IsFalse());
    EXPECT_THAT(reg.add("bar"), IsTrue());
    EXPECT_THAT(reg.add("baz"), IsTrue());
    EXPECT_THAT(reg.add("bar"), IsFalse());

    EXPECT_THAT(reg.subs(), ElementsAre("foo", "bar", "baz"));
}

//-------------------------------------------------------------------------
