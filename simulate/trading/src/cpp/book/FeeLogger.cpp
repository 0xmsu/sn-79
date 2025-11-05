/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "Simulation.hpp"
#include "util.hpp"
#include "taosim/book/FeeLogger.hpp"
#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>

//-------------------------------------------------------------------------

FeeLogger::FeeLogger(
    const fs::path &filepath, 
    std::chrono::system_clock::time_point startTimePoint,
    decltype(ExchangeSignals::feeLog) &signal,
    Simulation *simulation) noexcept
    : m_filepath{filepath},
      m_startTimePoint{startTimePoint},
      m_simulation{simulation},
      m_currentFilepath{filepath}
{
    m_timeConverter = taosim::simulation::timescaleToConverter(m_simulation->config().time().scale);

    m_logger = std::make_unique<spdlog::logger>("FeeLogger", makeFileSink());
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("%v");
    m_logger->trace(s_header);
    m_logger->flush();

    m_feed = signal.connect(
        [this](const FeePolicyWrapper* feePolicyWrapper, const taosim::FeeLogEvent& event) {
            log(feePolicyWrapper, event); 
        });
}

//-------------------------------------------------------------------------

void FeeLogger::log(const FeePolicyWrapper* feePolicyWrapper, const taosim::FeeLogEvent& event)
{
    updateSink();

    const auto time = m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp());

    const auto aggressingEntry = fmt::format(
        "{:%Y-%m-%d,%H:%M:%S},{},{},{},{},{},{},{}",
        time,
        event.aggressingAgentId,
        "Taker",
        event.fees.taker,
        feePolicyWrapper->getRates(event.bookId, event.aggressingAgentId).taker,
        event.price,
        event.volume,
        event.aggressingRatio);
    const auto restingEntry = fmt::format(
        "{:%Y-%m-%d,%H:%M:%S},{},{},{},{},{},{},{}",
        time,
        event.restingAgentId,
        "Maker",
        event.fees.maker,
        feePolicyWrapper->getRates(event.bookId, event.restingAgentId).maker,
        event.price,
        event.volume,
        event.restingRatio);

    m_logger->trace(aggressingEntry);
    m_logger->trace(restingEntry);
    m_logger->flush();
}

//-------------------------------------------------------------------------

void FeeLogger::updateSink()
{
    if (!m_simulation->logWindow()) {
        if (m_currentFilepath != m_filepath) [[unlikely]] {
            m_currentWindowBegin = taosim::simulation::kLogWindowMax;
            m_logger->sinks().clear();
            m_logger->sinks().push_back(makeFileSink());
            m_logger->set_pattern("%v");
            m_logger->trace(s_header);
            m_logger->flush();
        }
        return;
    }
    const auto end = std::min(
        m_currentWindowBegin + m_simulation->logWindow(), taosim::simulation::kLogWindowMax);
    const bool withinWindow = m_simulation->time().current < end;
    if (withinWindow) [[likely]] return;
    m_currentWindowBegin += m_simulation->logWindow();
    if (m_currentWindowBegin > taosim::simulation::kLogWindowMax) {
        m_currentWindowBegin = taosim::simulation::kLogWindowMax;
        m_simulation->logWindow() = {};
    }
    m_logger->sinks().clear();
    m_logger->sinks().push_back(makeFileSink());
    m_logger->set_pattern("%v");
    m_logger->trace(s_header);
    m_logger->flush();
}

//-------------------------------------------------------------------------

std::unique_ptr<spdlog::sinks::basic_file_sink_st> FeeLogger::makeFileSink()
{
    m_currentFilepath = [this] {
        if (!m_simulation->logWindow()) return m_filepath;
        return fs::path{fmt::format(
            "{}.{}-{}.log",
            (m_filepath.parent_path() / m_filepath.stem()).generic_string(),
            taosim::simulation::logFormatTime(m_timeConverter(m_currentWindowBegin)),
            taosim::simulation::logFormatTime(
                m_timeConverter(m_currentWindowBegin + m_simulation->logWindow())))};
    }();
    return std::make_unique<spdlog::sinks::basic_file_sink_st>(m_currentFilepath);
}

//-------------------------------------------------------------------------