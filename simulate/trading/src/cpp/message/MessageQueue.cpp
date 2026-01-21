/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/MessageQueue.hpp>

//-------------------------------------------------------------------------

namespace taosim::message
{

//-------------------------------------------------------------------------

MessageQueue::MessageQueue(std::vector<PrioritizedMessageWithId> messages) noexcept
    : m_queue{CompareQueueMessages{}, std::move(messages)}
{}

//-------------------------------------------------------------------------

bool MessageQueue::CompareQueueMessages::operator()(
    PrioritizedMessageWithId lhs, PrioritizedMessageWithId rhs)
{
    if (lhs.pmsg.marginCallId != rhs.pmsg.marginCallId) [[unlikely]] {
        return lhs.pmsg.marginCallId > rhs.pmsg.marginCallId;
    }
    if (lhs.pmsg.msg->arrival != rhs.pmsg.msg->arrival) [[likely]] {
        return lhs.pmsg.msg->arrival > rhs.pmsg.msg->arrival;
    }
    return lhs.id > rhs.id;
}

//-------------------------------------------------------------------------

}  // namespace taosim::message

//-------------------------------------------------------------------------
