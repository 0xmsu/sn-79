/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/TimeConfig.hpp>

#include <pugixml.hpp>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

class SimulationConfig
{
public:
    SimulationConfig() noexcept = default;
    SimulationConfig(TimeConfig time);

    [[nodiscard]] auto&& time(this auto&& self) noexcept
    {
        return std::forward_like<decltype(self)>(self.m_time);
    }

    [[nodiscard]] static SimulationConfig fromXML(pugi::xml_node node);

private:
    TimeConfig m_time;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
