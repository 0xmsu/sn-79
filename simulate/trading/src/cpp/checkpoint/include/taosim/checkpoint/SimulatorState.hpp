/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/SimulationManager.hpp>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

struct SimulatorState
{
    simulation::SimulationManager* mngr;
};

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------