/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/L2Logger.hpp>

#include "Simulation.hpp"
#include "common.hpp"

#include <fmt/chrono.h>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

L2Logger::L2Logger(
    const fs::path& filepath,
    uint32_t depth,
    std::chrono::system_clock::time_point startTimePoint,
    decltype(BookSignals::L2)& signal,
    Simulation* simulation) noexcept
    : logging::RotatingLoggerBase(logging::RotatingLoggerBaseDesc{
        .name = "L2Logger",
        .simulation = simulation,
        .filepath = filepath,
        .startTimePoint = startTimePoint,
        .header = std::string{L2Logger::s_header}
      }),
      m_depth{std::max(depth, 1u)},
      m_feed{signal.connect([this](const Book* book) { log(book); })}
{}

//-------------------------------------------------------------------------

void L2Logger::log(const Book* book)
{
    updateSink();
    const std::string newLog = createEntryAS(book);
    if (newLog != "" && newLog != m_lastLog) {
        m_logger->trace(newLog);
        m_logger->flush();
    }
    m_lastLog = newLog;
}

//-------------------------------------------------------------------------

std::string L2Logger::createEntryAS(const Book* book) const noexcept
{
    if (book->buyQueue().empty() || book->sellQueue().empty()) [[unlikely]] {
        return {};
    }

    auto levelFormatter = [](const auto& level) -> std::string {
        return fmt::format("({}@{})", level.volume(), level.price());
    };

    return fmt::format(
        // Date,Time,
        "{:%Y-%m-%d,%H:%M:%S},"
        // Symbol,Market,
        "S{:0{}}-SIMU,RAYX,"
        // BidVol,BidPrice,
        "{},{},"
        // AskVol,AskPrice,
        "{},{},"
        // QuoteCondition,Time,EndTime, (legacy)
        ",,,"
        // BidLevels,
        "{},"
        // AskLevels,
        "{},",
        m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp()),
        book->id(), 3,
        book->buyQueue().back().volume(), book->buyQueue().back().price(),
        book->sellQueue().front().volume(), book->sellQueue().front().price(),
        fmt::join(
            book->buyQueue()
            | views::reverse
            | views::take(m_depth)
            | views::reverse
            | views::transform(levelFormatter),
            " "),
        fmt::join(
            book->sellQueue()
            | views::take(m_depth)
            | views::transform(levelFormatter),
            " "));
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
