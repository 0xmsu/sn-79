/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Agent.hpp"
#include "DistributedProxyAgent.hpp"
#include "LocalAgentRoster.hpp"
#include "MultiBookExchangeAgent.hpp"
#include "PythonAgent.hpp"

#include <span>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

class LocalAgentManager
{
public:
    LocalAgentManager(Simulation* simulation) noexcept;

    [[nodiscard]] auto begin() { return m_agents.begin(); }
    [[nodiscard]] auto end() { return m_agents.end(); }

    void createAgentsInstanced(
        pugi::xml_node node,
        std::function<void(pugi::xml_node)> creationCallback = {});

    [[nodiscard]] auto&& agents(this auto&& self) noexcept { return self.m_agents; }
    [[nodiscard]] const std::unique_ptr<LocalAgentRoster>& roster() const noexcept { return m_roster; }

private:
    template<std::derived_from<Agent> T>
    void createAgentInstanced(pugi::xml_node node);

    Simulation* m_simulation;
    // Invariant: sorted.
    std::vector<std::unique_ptr<Agent>> m_agents;
    std::unique_ptr<LocalAgentRoster> m_roster;
};

//-------------------------------------------------------------------------

template<>
void LocalAgentManager::createAgentInstanced<DistributedProxyAgent>(pugi::xml_node node);

template<>
void LocalAgentManager::createAgentInstanced<MultiBookExchangeAgent>(pugi::xml_node node);

template<>
void LocalAgentManager::createAgentInstanced<PythonAgent>(pugi::xml_node node);

//-------------------------------------------------------------------------
