/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "OrderLogAgent.hpp"

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Simulation.hpp"


OrderLogAgent::OrderLogAgent(Simulation* simulation) : Agent(simulation)
{}

OrderLogAgent::OrderLogAgent(Simulation* simulation, const std::string& name)
    : Agent(simulation, name)
{}

void OrderLogAgent::receiveMessage(Message::Ptr messagePtr)
{
    const Timestamp currentTimestamp = simulation()->currentTimestamp();

    if (messagePtr->type == "EVENT_SIMULATION_START") {
        simulation()->dispatchMessage(
            currentTimestamp,
            0,
            name(),
            m_exchange,
            "SUBSCRIBE_EVENT_ORDER_LIMIT",
            std::make_shared<EmptyPayload>());
        simulation()->dispatchMessage(
            currentTimestamp,
            0,
            name(),
            m_exchange,
            "SUBSCRIBE_EVENT_ORDER_MARKET",
            std::make_shared<EmptyPayload>());
    }
    else if (messagePtr->type == "EVENT_ORDER_MARKET") {
        auto pptr = std::dynamic_pointer_cast<EventOrderMarketPayload>(messagePtr->payload);
        const auto& order = pptr->order;

        std::cout << name() << ": ";
        std::cout << taosim::json::json2str([&order] {
            rapidjson::Document json;
            order.jsonSerialize(json);
            return json;
        }());
        std::cout << std::endl;
    }
    else if (messagePtr->type == "EVENT_ORDER_LIMIT") {
        auto pptr = std::dynamic_pointer_cast<EventOrderLimitPayload>(messagePtr->payload);
        const auto& order = pptr->order;

        std::cout << name() << ": ";
        std::cout << taosim::json::json2str([&order] {
            rapidjson::Document json;
            order.jsonSerialize(json);
            return json;
        }());
        std::cout << std::endl;
    }
}

void OrderLogAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    pugi::xml_attribute att;
    if (!(att = node.attribute("exchange")).empty()) {
        m_exchange = "EXCHANGE";
    }
}