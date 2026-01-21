/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>
#include <rapidcsv.h>

#include <taosim/simulation/ISimulation.hpp>
#include "Process.hpp"

//-------------------------------------------------------------------------

namespace taosim::exchange
{

class ExchangeConfig;

}  // namespace taosim::exchange

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

class ProcessFactory
{
public:
    ProcessFactory(
        taosim::simulation::ISimulation* simulation,
        taosim::exchange::ExchangeConfig* exchangeConfig) noexcept;

    [[nodiscard]] std::unique_ptr<Process> createFromXML(pugi::xml_node node, uint64_t seedShift = 0);

private:
    taosim::simulation::ISimulation* m_simulation;
    taosim::exchange::ExchangeConfig* m_exchangeConfig;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------