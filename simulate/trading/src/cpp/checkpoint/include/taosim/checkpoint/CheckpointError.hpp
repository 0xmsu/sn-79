/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>

#include <stdexcept>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

struct CheckpointError : std::exception
{
    std::string message;

    CheckpointError(
        std::string_view msg = {},
        std::source_location sl = std::source_location::current()) noexcept
    {
        message = fmt::format(
            "Checkpoint error @ {}#L{}{}",
            sl.file_name(),
            sl.line(),
            msg.empty() ? "" : fmt::format(": {}", msg));
    }

    const char* what() const noexcept override { return message.c_str(); }
};

}  // taosim::checkpoint

//-------------------------------------------------------------------------
