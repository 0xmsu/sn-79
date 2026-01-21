/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/Fees.hpp>
#include "Trade.hpp"
#include "common.hpp"
#include <taosim/exchange/FeePolicy.hpp>

#include <boost/circular_buffer.hpp>
#include <unordered_set>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

struct Volumes
{
    decimal_t aggressive{};
    decimal_t passive{};
};

struct FeeMath {
    decimal_t maxMakerRate;
    decimal_t maxTakerRate;
    decimal_t targetMTR;
    decimal_t shapeMakerFee;
    decimal_t shapeMakerRebate;
    decimal_t coefLHS;
    decimal_t coefRHS;

    FeeMath(decimal_t maxRate_1, decimal_t maxRate_2, decimal_t targetMTR_,
        decimal_t shape_1, decimal_t shape_2)
        : maxMakerRate{maxRate_1}, maxTakerRate{maxRate_2}, 
        targetMTR{targetMTR_}, shapeMakerFee{shape_1}, shapeMakerRebate{shape_2}
    {
        coefLHS = -(maxMakerRate / util::pow(targetMTR, shapeMakerFee));
        coefRHS = maxTakerRate / util::pow(1_dec - targetMTR, shapeMakerRebate);
    }

    [[nodiscard]] decimal_t calculate(decimal_t x) const noexcept
    {
        if (util::round(targetMTR - x,3) == 0_dec) return {};
        if (targetMTR > x) return dyLHSFunc(x);
        return dyRHSFunc(x);
    }

    [[nodiscard]] decimal_t dyLHSFunc(decimal_t x) const noexcept
    {
        decimal_t out = coefLHS * util::pow(targetMTR - x, shapeMakerFee);
        return out;
    }
    
    [[nodiscard]] decimal_t dyRHSFunc(decimal_t x) const noexcept
    {
        decimal_t out =  coefRHS * util::pow(x - targetMTR, shapeMakerRebate);
        return out;
    }
};

struct DFPolicyDesc
{
    Simulation* simulation;
    int historySlots;
    decimal_t makerFee;
    decimal_t takerFee;
    decimal_t maxMakerRate;
    decimal_t maxTakerRate;
    decimal_t targetMTR;
    decimal_t shapeMakerFee;
    decimal_t shapeMakerRebate;
};

//-------------------------------------------------------------------------

class DynamicFeePolicy : public FeePolicy
{
public:
    explicit DynamicFeePolicy(const DFPolicyDesc& desc);

    [[nodiscard]] Fees calculateFees(const TradeDesc& tradeDesc) const override;
    [[nodiscard]] Fees getRates(BookId bookId, AgentId agentId) const override;
    [[nodiscard]] decimal_t makerTakerRatio(BookId bookId) const;

    void updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, 
        std::optional<bool> isAggressive = {}) override;
    void resetHistory() noexcept override;
    void resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept override;
    
    [[nodiscard]] static std::unique_ptr<DynamicFeePolicy> fromXML(
        pugi::xml_node node, Simulation* simulation);
        
    [[nodiscard]] auto&& lastUpdate(this auto&& self) noexcept { return self.m_lastUpdate; }
    [[nodiscard]] auto&& totalVolumes(this auto&& self) noexcept { return self.m_totalVolumes; }
    [[nodiscard]] auto&& totalVolumesPrev(this auto&& self) noexcept { return self.m_totalVolumesPrev; }
    [[nodiscard]] auto&& volumes(this auto&& self) noexcept { return self.m_volumes; }
        
protected:
    Simulation* m_simulation;
    FeeMath m_feeMath;
    int m_historySlots;
    decimal_t m_makerFee;
    decimal_t m_takerFee;
    Timestamp m_lastUpdate;
    std::map<BookId, Volumes> m_totalVolumes;
    std::map<BookId, Volumes> m_totalVolumesPrev;
    std::map<BookId, boost::circular_buffer<Volumes>> m_volumes;
    
    [[nodiscard]] Fees dynamicRates(BookId bookId, AgentId agentId) const noexcept;
    [[nodiscard]] Fees exchangeRates(BookId bookId, AgentId agentId) const noexcept;
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
