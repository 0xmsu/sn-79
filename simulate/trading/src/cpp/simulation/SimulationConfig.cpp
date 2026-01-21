/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/SimulationConfig.hpp>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

SimulationConfig SimulationConfig::fromXML(pugi::xml_node node)
{
    return SimulationConfig{TimeConfig::fromXML(node)};
}

//-------------------------------------------------------------------------

SimulationConfig::SimulationConfig(TimeConfig time)
{
    m_time = time;
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
