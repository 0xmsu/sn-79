/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/logging/RotatingLoggerBase.hpp>

#include "BookSignals.hpp"

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class L2Logger final : public logging::RotatingLoggerBase
{
public:
    L2Logger(
        const fs::path& filepath,
        uint32_t depth,
        std::chrono::system_clock::time_point startTimePoint,
        decltype(BookSignals::L2)& signal,
        Simulation* simulation) noexcept;

    static constexpr std::string_view s_header =
        "Date,Time,Symbol,Market,BidVol,BidPrice,AskVol,AskPrice,"
        "QuoteCondition,Time,EndTime,BidLevels,AskLevels";

private:
    void log(const Book* book);

    [[nodiscard]] std::string createEntryAS(const Book* book) const noexcept;

    boost::signals2::scoped_connection m_feed;
    uint32_t m_depth;
    std::string m_lastLog;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
