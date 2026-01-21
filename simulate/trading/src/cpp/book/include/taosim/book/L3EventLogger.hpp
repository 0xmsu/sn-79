/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/logging/RotatingLoggerBase.hpp>
#include <taosim/exchange/ExchangeSignals.hpp>
#include "JsonSerializable.hpp"

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class L3EventLogger final : public logging::RotatingLoggerBase
{
public:
    L3EventLogger(
        const fs::path& filepath,
        std::chrono::system_clock::time_point startTimePoint,
        decltype(exchange::ExchangeSignals::L3)& signal,
        Simulation* simulation) noexcept;

    static constexpr std::string_view s_header = "date,time,event";

private:
    void log(taosim::L3LogEvent event);

    bs2::scoped_connection m_feed;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
