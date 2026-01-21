/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/TimeConfig.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string_view>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

struct RotatingLoggerBaseDesc
{
    std::string name;
    Simulation* simulation;
    std::filesystem::path filepath;
    std::chrono::system_clock::time_point startTimePoint;
    std::string header;
};

struct FileSinkWithInfo
{
    std::unique_ptr<spdlog::sinks::basic_file_sink_st> sink;
    bool fileExisted;
};

//-------------------------------------------------------------------------

class RotatingLoggerBase
{
public:
    explicit RotatingLoggerBase(const RotatingLoggerBaseDesc& desc) noexcept;

    [[nodiscard]] const std::filesystem::path& filepath() const noexcept { return m_filepath; }

protected:
    [[nodiscard]] FileSinkWithInfo makeFileSink();
    void updateSink();

    std::unique_ptr<spdlog::logger> m_logger;
    Simulation* m_simulation;
    std::filesystem::path m_filepath;
    std::chrono::system_clock::time_point m_startTimePoint;
    std::filesystem::path m_currentFilepath;
    simulation::TimestampConversionFn m_timeConverter;
    Timestamp m_currentWindowBegin;
    std::string m_header;
};

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------