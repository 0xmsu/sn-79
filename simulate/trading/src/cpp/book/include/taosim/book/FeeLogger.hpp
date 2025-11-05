/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "JsonSerializable.hpp"
#include "taosim/simulation/TimeConfig.hpp"
#include "ExchangeSignals.hpp"
#include "FeeLogEvent.hpp"
#include "taosim/exchange/FeePolicyWrapper.hpp"

#include "util.hpp"

#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

class FeeLogger
{
public:
    FeeLogger(
        const fs::path& filepath,
        std::chrono::system_clock::time_point startTimePoint,
        decltype(ExchangeSignals::feeLog)& signal,
        Simulation* simulation) noexcept;

    [[nodiscard]] const fs::path& filepath() const noexcept { return m_filepath; }

    static constexpr std::string_view s_header = "Date,Time,AgentId,Role,Fee,FeeRate,Price,Volume,FeeRatio";

private:
    void log(const FeePolicyWrapper* feePolicyWrapper, const taosim::FeeLogEvent& event);
    void updateSink();

    [[nodiscard]] std::unique_ptr<spdlog::sinks::basic_file_sink_st> makeFileSink();
    
    std::unique_ptr<spdlog::logger> m_logger;
    fs::path m_filepath;
    std::chrono::system_clock::time_point m_startTimePoint;
    bs2::scoped_connection m_feed;
    Simulation* m_simulation;
    taosim::simulation::TimestampConversionFn m_timeConverter{};
    Timestamp m_currentWindowBegin{};
    fs::path m_currentFilepath;
};

//-------------------------------------------------------------------------
