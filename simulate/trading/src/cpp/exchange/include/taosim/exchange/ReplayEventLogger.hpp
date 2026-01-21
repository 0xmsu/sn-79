/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/logging/RotatingLoggerBase.hpp>
#include <taosim/message/Message.hpp>
#include <taosim/simulation/TimeConfig.hpp>

#include <string_view>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

class ReplayEventLogger : public logging::RotatingLoggerBase
{
public:
    ReplayEventLogger(
        const fs::path& filepath,
        std::chrono::system_clock::time_point startTimePoint,
        Simulation* simulation) noexcept;

    void log(Message::Ptr event);

    static constexpr std::string_view s_header = "date,time,message";

private:
    [[nodiscard]] rapidjson::Document makeLogEntryJson(Message::Ptr msg);
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
