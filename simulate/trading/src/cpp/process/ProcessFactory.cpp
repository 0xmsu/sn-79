/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ProcessFactory.hpp"

#include "GBM.hpp"
#include "FundamentalPrice.hpp"
#include "JumpDiffusion.hpp"
#include "FuturesSignal.hpp"
#include "Simulation.hpp"
#include "taosim/exchange/ExchangeConfig.hpp"

//-------------------------------------------------------------------------

ProcessFactory::ProcessFactory(
    taosim::simulation::ISimulation* simulation, taosim::exchange::ExchangeConfig* exchangeConfig) noexcept
    : m_simulation{simulation}, m_exchangeConfig{exchangeConfig}
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
            taosim::util::decimal2double(m_exchangeConfig->initialPrice));
    }
    else if (name == "JumpDiffusion") {
        return JumpDiffusion::fromXML(node, seedShift);
    }
    else if (name == "FuturesSignal") {
        return FuturesSignal::fromXML(
            m_simulation, 
            node, 
            seedShift, 
            taosim::util::decimal2double(m_exchangeConfig->initialPrice));
    }

    throw std::invalid_argument(fmt::format(
        "{}: Unknown Process type {}", std::source_location::current().function_name(), name));
}

//-------------------------------------------------------------------------

std::unique_ptr<Process> ProcessFactory::createFromCheckpoint(const rapidjson::Value& json)
{
    std::string_view name = json["name"].GetString();

    if (name == "GBM") {
        return GBM::fromCheckpoint(json);
    } else if (name == "FundamentalPrice") {
        return FundamentalPrice::fromCheckpoint(
            m_simulation, json, taosim::util::decimal2double(m_exchangeConfig->initialPrice));
    }

    throw std::invalid_argument(fmt::format(
        "{}: Unknown Process type {}", std::source_location::current().function_name(), name));
}

//-------------------------------------------------------------------------