/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/Fees.hpp>
#include "Trade.hpp"
#include "common.hpp"

#include <unordered_set>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

struct TradeDesc
{
    BookId bookId;
    AgentId restingAgentId;
    AgentId aggressingAgentId;
    Trade::Ptr trade;
};

//-------------------------------------------------------------------------

class FeePolicy
{
public:
    virtual ~FeePolicy() = default;

    virtual Fees calculateFees(const TradeDesc& tradeDesc) const = 0;
    // virtual void updateAgentsTiers() noexcept = 0;
    virtual void updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, 
        std::optional<bool> isAggressive = {}) = 0;
    virtual void resetHistory() noexcept = 0;
    virtual void resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept = 0;
    virtual Fees getRates(BookId bookId, AgentId agentId) const = 0;
    // virtual decimal_t agentVolumes(this auto&& self) noexcept = 0;
    
    // [[nodiscard]] static std::unique_ptr<FeePolicy> fromXML(
    //     pugi::xml_node node, Simulation* simulation);

    static decimal_t checkFeeRate(double feeRate)
    {
        static constexpr double feeRateMin{-1.0}, feeRateMax{1.0};

        if (!(feeRateMin < feeRate && feeRate < feeRateMax)) {
            throw std::invalid_argument{fmt::format(
                "{}: Fee should be between {} and {}; was {}",
                std::source_location::current().function_name(),
                feeRateMin, feeRateMax, feeRate)};
        }

        return decimal_t{feeRate};
    }

};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
