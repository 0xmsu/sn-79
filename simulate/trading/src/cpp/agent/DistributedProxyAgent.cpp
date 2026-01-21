/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "DistributedProxyAgent.hpp"

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Simulation.hpp"
#include "json_util.hpp"
#include "util.hpp"

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <source_location>
#include <chrono>
#include <thread>

//-------------------------------------------------------------------------

DistributedProxyAgent::DistributedProxyAgent(Simulation* simulation)
    : Agent{simulation, "DISTRIBUTED_PROXY_AGENT"}
{}

//-------------------------------------------------------------------------

void DistributedProxyAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "MULTIBOOK_STATE_PUBLISH") {
        return;
    } else if (msg->type == "EVENT_SIMULATION_START") {
        return;
    }
    m_messages.push_back(msg);
}

//-------------------------------------------------------------------------

void DistributedProxyAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);
}

//-------------------------------------------------------------------------
