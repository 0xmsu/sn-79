/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstdint>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

enum class SimulationState : uint32_t
{
    INACTIVE,
    STARTED,
    STOPPED
};

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
