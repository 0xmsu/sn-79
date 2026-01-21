/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/MessageQueue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim::message;

//-------------------------------------------------------------------------

TEST(MessageQueueTest, API)
{
    MessageQueue messageQueue;

    EXPECT_TRUE(messageQueue.empty());
    EXPECT_EQ(messageQueue.size(), 0);

    static constexpr int pushCount = 4;
    for (int i = 0; i < pushCount; ++i) {
        messageQueue.push(
            Message::create(0, 0, "foo", "bar", "baz", MessagePayload::create<EmptyPayload>()));
    }

    EXPECT_FALSE(messageQueue.empty());
    EXPECT_EQ(messageQueue.size(), pushCount);

    static constexpr int popCount = 3;
    for (int i = 0; i < popCount; ++i) {
        messageQueue.pop();
    }

    EXPECT_FALSE(messageQueue.empty());
    EXPECT_EQ(messageQueue.size(), pushCount - popCount);
}

//-------------------------------------------------------------------------

TEST(MessageQueueTest, EqualArrivals)
{
    MessageQueue messageQueue;

    for (auto testId : {"1st", "2nd", "3rd", "4th"}) {
        messageQueue.push(
            Message::create(0, 0, "foo", "bar", testId, MessagePayload::create<EmptyPayload>()));
    }

    std::vector<std::string> poppedTestIds;
    for (int i = 0; i < 3; ++i) {
        poppedTestIds.push_back(messageQueue.top()->type);
        messageQueue.pop();
    }

    EXPECT_THAT(poppedTestIds, testing::ElementsAre("1st", "2nd", "3rd"));
}

//-------------------------------------------------------------------------

TEST(MessageQueueTest, DifferingArrivals)
{
    MessageQueue messageQueue;

    for (auto testId : {"1st", "2nd", "3rd", "4th"}) {
        messageQueue.push(
            Message::create(
                0,
                Timestamp{4} - (testId[0] - '0'),
                "foo",
                "bar",
                testId,
                MessagePayload::create<EmptyPayload>()));
    }

    std::vector<std::string> poppedTestIds;
    for (int i = 0; i < 3; ++i) {
        poppedTestIds.push_back(messageQueue.top()->type);
        messageQueue.pop();
    }

    EXPECT_THAT(poppedTestIds, testing::ElementsAre("4th", "3rd", "2nd"));
}

//-------------------------------------------------------------------------

TEST(MessageQueueTest, MarginCallPriority)
{
    MessageQueue messageQueue;

    for (auto testId : {"1st", "2nd", "3rd", "4th"}) {
        messageQueue.push(PrioritizedMessage(
            Message::create(0, 0, "foo", "bar", testId, MessagePayload::create<EmptyPayload>()),
            4 - (testId[0] - '0')));
    }

    std::vector<std::string> poppedTestIds;
    do {
        poppedTestIds.push_back(messageQueue.top()->type);
        messageQueue.pop();
    } while (!messageQueue.empty());

    EXPECT_THAT(poppedTestIds, testing::ElementsAre("4th", "3rd", "2nd", "1st"));
}

//-------------------------------------------------------------------------
