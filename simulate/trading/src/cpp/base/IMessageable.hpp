/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/Message.hpp>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

class IMessageable
{
public:
    const std::string& name() const noexcept { return m_name; }
    Simulation* simulation() const noexcept { return m_simulation; }

    virtual void receiveMessage(Message::Ptr msg) = 0;

    virtual void respondToMessage(
        Message::Ptr msg,
        const std::string& typePrefix,
        MessagePayload::Ptr payload,
        Timestamp processingDelay = 0) const;

    virtual void respondToMessage(
        Message::Ptr msg, MessagePayload::Ptr payload, Timestamp processingDelay = 0) const;

    virtual void fastRespondToMessage(
        Message::Ptr msg,
        const std::string& typePrefix,
        MessagePayload::Ptr payload,
        Timestamp processingDelay = 0) const;

    virtual void fastRespondToMessage(
        Message::Ptr msg, MessagePayload::Ptr payload, Timestamp processingDelay = 0) const;

protected:
    IMessageable(Simulation* simulation, const std::string& name) noexcept
        : m_simulation{simulation}, m_name{name}
    {}

    virtual ~IMessageable() = default;

    void setName(const std::string& name) noexcept { m_name = name; }

private:
    Simulation* m_simulation;
    std::string m_name;
};

//-------------------------------------------------------------------------
