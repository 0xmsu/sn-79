/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Agent.hpp"
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <vector>

//-------------------------------------------------------------------------

class DistributedProxyAgent : public Agent
{
public:
    DistributedProxyAgent(Simulation* simulation);

    [[nodiscard]] std::span<Message::Ptr> messages() noexcept { return m_messages; }

    void clearMessages() noexcept { m_messages.clear(); };

    virtual void receiveMessage(Message::Ptr msg) override;
    virtual void configure(const pugi::xml_node& node) override;

private:
    std::vector<Message::Ptr> m_messages;
};

//-------------------------------------------------------------------------
