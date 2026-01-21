/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/logging/RotatingLoggerBase.hpp>
#include <taosim/exchange/FeePolicyWrapper.hpp>
#include <taosim/exchange/ExchangeSignals.hpp>
#include "FeeLogEvent.hpp"

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class FeeLogger final : public logging::RotatingLoggerBase
{
public:
    FeeLogger(
        const fs::path& filepath,
        std::chrono::system_clock::time_point startTimePoint,
        decltype(exchange::ExchangeSignals::feeLog)& signal,
        Simulation* simulation) noexcept;

    static constexpr std::string_view s_header =
        "Date,Time,AgentId,Role,Fee,FeeRate,Price,Volume,FeeRatio";

private:
    void log(const exchange::FeePolicyWrapper* feePolicyWrapper, const FeeLogEvent& event);
    
    bs2::scoped_connection m_feed;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
