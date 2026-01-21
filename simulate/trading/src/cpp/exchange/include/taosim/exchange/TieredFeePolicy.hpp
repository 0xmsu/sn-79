/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/exchange/Fees.hpp>
#include "Trade.hpp"
#include "common.hpp"
#include <taosim/exchange/FeePolicy.hpp>

#include <unordered_set>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

struct Tier
{
    decimal_t volumeRequired;
    decimal_t makerFeeRate;
    decimal_t takerFeeRate;
};

struct TFPolicyDesc
{
    Simulation* simulation;
    int historySlots;
    Timestamp slotPeriod;
    std::vector<Tier> tiers;
};

//-------------------------------------------------------------------------

class TieredFeePolicy : public FeePolicy
{
public:
    explicit TieredFeePolicy(const TFPolicyDesc& desc);

    [[nodiscard]] auto&& agentTiers(this auto&& self) noexcept { return self.m_agentTiers; }
    [[nodiscard]] auto&& agentVolumes(this auto&& self) noexcept { return self.m_agentVolumes; }

    [[nodiscard]] Fees getRates(BookId bookId, AgentId agentId) const override;
    [[nodiscard]] Fees calculateFees(const TradeDesc& tradeDesc) const override;
    [[nodiscard]] int historySlots() const noexcept { return m_historySlots; }
    [[nodiscard]] Timestamp slotPeriod() const noexcept { return m_slotPeriod; }
    [[nodiscard]] std::span<const Tier> tiers() const noexcept { return m_tiers; }

    void updateAgentsTiers() noexcept;
    void updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume,
        std::optional<bool> isAggressive = {}) override;
    void resetHistory() noexcept override;
    void resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept override;
    
    [[nodiscard]] static std::unique_ptr<TieredFeePolicy> fromXML(
        pugi::xml_node node, Simulation* simulation);
        
protected:
    using TierIdx = int32_t;

    [[nodiscard]] const Tier& findTierForVolume(decimal_t volume) const noexcept;
    [[nodiscard]] const Tier& findTierForAgent(BookId bookId, AgentId agentId) const noexcept;

    Simulation* m_simulation;
    int m_historySlots;
    Timestamp m_slotPeriod;
    std::vector<Tier> m_tiers;
    std::map<AgentId, std::map<BookId, TierIdx>> m_agentTiers;
    std::map<AgentId, std::map<BookId, std::vector<decimal_t>>> m_agentVolumes;
};

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
