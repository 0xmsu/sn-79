/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/logging/RotatingLoggerBase.hpp>

#include "Simulation.hpp"

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

RotatingLoggerBase::RotatingLoggerBase(const RotatingLoggerBaseDesc& desc) noexcept
    : m_simulation{desc.simulation},
      m_filepath{desc.filepath},
      m_startTimePoint{desc.startTimePoint},
      m_header{desc.header}
{
    m_timeConverter = simulation::timescaleToConverter(m_simulation->config().time().scale);

    m_currentWindowBegin = m_simulation->logWindow()
        ? m_simulation->currentTimestamp() / m_simulation->logWindow() * m_simulation->logWindow()
        : taosim::simulation::kLogWindowMax;

    auto [sink, fileExisted] = makeFileSink();

    m_logger = std::make_unique<spdlog::logger>(desc.name, std::move(sink));
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("%v");

    if (!fileExisted) {
        m_logger->trace(m_header);
        m_logger->flush();
    }
}

//-------------------------------------------------------------------------

void RotatingLoggerBase::updateSink()
{
    if (!m_simulation->logWindow()) return;

    auto update = [&] {
        m_logger->sinks().clear();
        m_logger->sinks().push_back(makeFileSink().sink);
        m_logger->set_pattern("%v");
        m_logger->trace(m_header);
        m_logger->flush();
    };

    const auto end = std::min(
        m_currentWindowBegin + m_simulation->logWindow(), taosim::simulation::kLogWindowMax);

    const bool withinWindow = m_simulation->currentTimestamp() < end;

    if (withinWindow) [[likely]] return;

    m_currentWindowBegin += m_simulation->logWindow();

    if (m_currentWindowBegin > taosim::simulation::kLogWindowMax) {
        m_currentWindowBegin = taosim::simulation::kLogWindowMax;
        m_simulation->logWindow() = {};
    }

    update();
}

//-------------------------------------------------------------------------

FileSinkWithInfo RotatingLoggerBase::makeFileSink()
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

    const bool fileExisted = fs::exists(m_currentFilepath);

    return {
        .sink = std::make_unique<spdlog::sinks::basic_file_sink_st>(m_currentFilepath),
        .fileExisted = fileExisted
    };
}

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------
