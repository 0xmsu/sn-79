/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/ProcessFactory.hpp>

#include <taosim/exchange/ExchangeConfig.hpp>
#include <taosim/process/GBM.hpp>
#include <taosim/process/FundamentalPrice.hpp>
#include <taosim/process/FuturesSignal.hpp>
#include <taosim/process/JumpDiffusion.hpp>
#include <taosim/process/MagneticField.hpp>
#include <Simulation.hpp>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

ProcessFactory::ProcessFactory(
    taosim::simulation::ISimulation* simulation,
    taosim::exchange::ExchangeConfig* exchangeConfig) noexcept
    : m_simulation{simulation},
      m_exchangeConfig{exchangeConfig}
{}

//-------------------------------------------------------------------------

std::unique_ptr<Process> ProcessFactory::createFromXML(pugi::xml_node node, uint64_t seedShift)
{
    std::string_view name = node.name();

    if (name == "GBM") {
        return GBM::fromXML(node, seedShift);
    }
    else if (name == "FundamentalPrice") {
        return FundamentalPrice::fromXML(
            m_simulation,
            node,
            seedShift,
            taosim::util::decimal2double(m_exchangeConfig->initialPrice)
        );
    }
    else if (name == "JumpDiffusion") {
        return JumpDiffusion::fromXML(node, seedShift);
    }
    else if (name == "FuturesSignal") {
        return FuturesSignal::fromXML(
            m_simulation, 
            node, 
            seedShift, 
            taosim::util::decimal2double(m_exchangeConfig->initialPrice)
        );
    } else if (name == "MagneticField") {
        return MagneticField::fromXML(node, m_simulation, seedShift);
    }

    throw std::invalid_argument(fmt::format(
        "{}: Unknown Process type {}", std::source_location::current().function_name(), name));
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------