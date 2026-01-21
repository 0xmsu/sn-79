/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Simulation.hpp"

#include <taosim/exchange/DynamicFeePolicy.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::exchange
{

//-------------------------------------------------------------------------

taosim::exchange::DynamicFeePolicy::DynamicFeePolicy(const DFPolicyDesc &desc)
    : m_simulation{desc.simulation},
      m_historySlots{desc.historySlots},
      m_makerFee{desc.makerFee},
      m_takerFee{desc.takerFee},
      m_feeMath{desc.maxMakerRate, desc.maxTakerRate, desc.targetMTR, desc.shapeMakerFee, desc.shapeMakerRebate},
      m_lastUpdate{0}
{}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::calculateFees(const TradeDesc &tradeDesc) const
{
    [[maybe_unused]] const auto [bookId, restingAgentId, aggressingAgentId, trade] = tradeDesc;
    const auto volumeWeightedPrice = trade->volume() * trade->price();
    return {
        .maker = getRates(bookId, restingAgentId).maker * volumeWeightedPrice,
        .taker = getRates(bookId, aggressingAgentId).taker * volumeWeightedPrice
    };
}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::getRates(BookId bookId, AgentId agentId) const
{    
    return exchangeRates(bookId, agentId) + dynamicRates(bookId, agentId);
}

//-------------------------------------------------------------------------

decimal_t DynamicFeePolicy::makerTakerRatio(BookId bookId) const
{
    auto it = m_totalVolumesPrev.find(bookId);
    if (it != m_totalVolumesPrev.end()){
        if (m_volumes.at(bookId).size() < m_historySlots)
            return m_feeMath.targetMTR;
        if (it->second.aggressive + it->second.passive == 0_dec){
            return m_feeMath.targetMTR;
        }
        return it->second.passive / (it->second.passive + it->second.aggressive);
    }
    return m_feeMath.targetMTR;
} 

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::dynamicRates(BookId bookId, AgentId agentId) const noexcept
{
    decimal_t x = makerTakerRatio(bookId);

    if (x == 0_dec) return {};

    return {
        .maker = m_feeMath.calculate(x),
        .taker = -m_feeMath.calculate(x)
    };
}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::exchangeRates(BookId bookId, AgentId agentId) const noexcept
{
    return {
        .maker = m_makerFee,
        .taker = m_takerFee
    };
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::updateHistory(
    Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, std::optional<bool> isAggressive)
{
    if (agentId < 0) return;

    if (!isAggressive.has_value()){
        throw std::runtime_error(fmt::format(
            "Book {}, agent {}: Dynamic fee policy must define if the agent is aggressive or not", bookId, agentId
        ));
    }

    auto& bookVolumes = m_totalVolumes[bookId];
    if (m_lastUpdate != timestamp && m_lastUpdate != 0){
        m_totalVolumesPrev[bookId] = bookVolumes;
    }

    if (isAggressive.value()){
        bookVolumes.aggressive += volume;
    } else {
        bookVolumes.passive += volume;
    }

    auto it = m_volumes.find(bookId);
    if (it != m_volumes.end()) {
        if (it->second.full() && m_lastUpdate != timestamp) {
            auto& oldest = it->second.front();
            bookVolumes.aggressive -= oldest.aggressive;
            bookVolumes.passive -= oldest.passive;
        }
        if (isAggressive.value()){
            if (m_lastUpdate == timestamp){
                it->second.back().aggressive += volume;
            } else {
                it->second.push_back(Volumes{volume, 0_dec});
            }
        } else {
            if (m_lastUpdate == timestamp){
                it->second.back().passive += volume;
            } else {
                it->second.push_back(Volumes{0_dec, volume});
            }
        }
    } else {
        m_volumes[bookId] = boost::circular_buffer<Volumes>(m_historySlots);
        if (isAggressive.value()){
            if (m_lastUpdate == timestamp){
                m_volumes[bookId].back().aggressive += volume;
            } else {
                m_volumes[bookId].push_back(Volumes{volume, 0_dec});
            }
        } else {
            if (m_lastUpdate == timestamp){
                m_volumes[bookId].back().passive += volume;
            } else {
                m_volumes[bookId].push_back(Volumes{0_dec, volume});
            }
        }
        
    }

    m_lastUpdate = timestamp;
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::resetHistory() noexcept
{
    for (auto& [bookId, vol_hist]: m_volumes) {
        vol_hist.clear();
    }
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept
{}

//-------------------------------------------------------------------------

std::unique_ptr<DynamicFeePolicy> DynamicFeePolicy::fromXML(pugi::xml_node node, Simulation *simulation)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getAttr = [](pugi::xml_node node, const char* name) {
        if (pugi::xml_attribute attr = node.attribute(name)) {
            return attr;
        }
        throw std::invalid_argument{fmt::format(
            "{}: Missing required argument '{}'", ctx, name)};
    };

    return std::make_unique<DynamicFeePolicy>(DFPolicyDesc{
        .simulation = simulation,
        .historySlots = getAttr(node, "historySlots").as_int(),
        .makerFee = checkFeeRate(getAttr(node, "makerFee").as_double()),
        .takerFee = checkFeeRate(getAttr(node, "takerFee").as_double()),
        .maxMakerRate = util::double2decimal(getAttr(node, "maxMakerRate").as_double()),
        .maxTakerRate = util::double2decimal(getAttr(node, "maxTakerRate").as_double()),
        .targetMTR = util::double2decimal(getAttr(node, "targetMTR").as_double()),
        .shapeMakerFee = util::double2decimal(getAttr(node, "shapeMakerFee").as_double()),
        .shapeMakerRebate = util::double2decimal(getAttr(node, "shapeMakerRebate").as_double())
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::exchange

//-------------------------------------------------------------------------