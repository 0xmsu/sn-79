/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>

#include <source_location>
#include <stdexcept>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

struct SimulationError : std::exception
{
    std::string message;

    SimulationError(
        std::string_view msg = {},
        std::source_location sl = std::source_location::current()) noexcept
    {
        message = fmt::format(
            "Simulation error @ {}#L{}{}",
            sl.file_name(),
            sl.line(),
            msg.empty() ? "" : fmt::format(": {}", msg));
    }

    const char* what() const noexcept override { return message.c_str(); }
};

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
