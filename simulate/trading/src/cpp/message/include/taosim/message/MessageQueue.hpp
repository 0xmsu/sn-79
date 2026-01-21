/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/dsa/PriorityQueue.hpp>
#include <taosim/message/Message.hpp>

#include <limits>
#include <queue>

//-------------------------------------------------------------------------

namespace taosim::message
{

//-------------------------------------------------------------------------

struct PrioritizedMessage
{
    Message::Ptr msg;
    uint64_t marginCallId;

    PrioritizedMessage() noexcept = default;
    PrioritizedMessage(
        Message::Ptr msg, uint64_t marginCallId = std::numeric_limits<uint64_t>::max()) noexcept
        : msg{msg}, marginCallId{marginCallId}
    {}
};

struct PrioritizedMessageWithId
{
    PrioritizedMessage pmsg;
    uint64_t id;

    PrioritizedMessageWithId() noexcept = default;
    PrioritizedMessageWithId(PrioritizedMessage pmsg, uint64_t id) noexcept
        : pmsg{pmsg}, id{id}
    {}
};

//-------------------------------------------------------------------------

class MessageQueue
{
public:
    MessageQueue() noexcept = default;
    explicit MessageQueue(std::vector<PrioritizedMessageWithId> messages) noexcept;

    [[nodiscard]] Message::Ptr top() const { return m_queue.top().pmsg.msg; }
    [[nodiscard]] bool empty() const { return m_queue.empty(); }
    [[nodiscard]] size_t size() const { return m_queue.size(); }

    void push(PrioritizedMessage pmsg) { m_queue.emplace(pmsg, m_idCounter++); }
    void pop() { m_queue.pop(); }
    void clear() { m_queue.clear(); }

    [[nodiscard]] auto&& queue(this auto&& self) noexcept { return self.m_queue; }
    [[nodiscard]] auto&& idCounter(this auto&& self) noexcept { return self.m_idCounter; }

private:
    struct CompareQueueMessages
    {
        bool operator()(PrioritizedMessageWithId lhs, PrioritizedMessageWithId rhs);
    };

    [[nodiscard]] const PrioritizedMessageWithId& prioTop() const { return m_queue.top(); }

    void push(PrioritizedMessageWithId pmsgWithId) { m_queue.push(pmsgWithId); }

    dsa::PriorityQueue<
        PrioritizedMessageWithId,
        std::vector<PrioritizedMessageWithId>,
        CompareQueueMessages> m_queue;
    uint64_t m_idCounter{};
};

//-------------------------------------------------------------------------

}  // namespace taosim::message

//-------------------------------------------------------------------------
