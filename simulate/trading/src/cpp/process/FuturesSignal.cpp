/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/FuturesSignal.hpp>

#include "Simulation.hpp"

#include <cmath>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

FuturesSignal::FuturesSignal(const FuturesSignalDesc& desc) noexcept
    : m_simulation{desc.simulation},
      m_bookId{desc.bookId},
      m_seedInterval{desc.seedInterval},
      m_lambda{desc.lambda}
{
    m_state.value = desc.X0;
    m_updatePeriod = desc.proc.updatePeriod;
    m_seedfile = (m_simulation->logDir() / "external_seed_sampled.csv").generic_string();
}

//-------------------------------------------------------------------------

double FuturesSignal::volumeFactor() noexcept
{
    ++m_state.factorCounter;
    m_state.volumeFactor = m_state.volumeFactor * std::exp(-m_lambda * m_state.factorCounter);
    return m_state.volumeFactor;
}

//-------------------------------------------------------------------------

void FuturesSignal::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        if (timestamp - m_state.lastSeedTime >= m_seedInterval) {
            if ( fs::exists( m_seedfile ) ) {
                int count = m_state.lastCount;
                float seed = 0.0;
                try {
                    std::vector<std::string> lines = taosim::util::getLastLines(m_seedfile, 2);
                    if (lines.size() >= 2) {
                        std::vector<std::string> line = taosim::util::split(lines[lines.size() - 2],',');
                        if (line.size() >= 2) {
                            count = std::stoi(line[0]);
                            seed = std::stof(line[1]);
                            if (auto simulation = dynamic_cast<Simulation*>(m_simulation)) {
                                simulation->logDebug("FuturesSignal::update : READ {}", lines[lines.size() - 2]);
                            }
                        } else {
                            fmt::println("FuturesSignal::update : FAILED TO GET SEED FROM LINE - {}", lines[lines.size() - 2]);
                        }
                    } else {
                        if (m_state.lastCount > 0) {
                            fmt::println("FuturesSignal::update : FAILED TO GET SEED FROM FILE - NO DATA ({} LINES READ)", lines.size());
                        }                    
                    }
                } catch (const std::exception &exc) {
                    fmt::println("FuturesSignal::update : ERROR GETTING SEED FROM FILE - {}", exc.what());
                }
                if (count != m_state.lastCount) {
                    if (m_state.value> 0.0) {
                        m_state.logReturn = std::log(seed / m_state.value);
                        m_state.volumeFactor = std::min(2.0, std::exp(std::abs(m_state.logReturn)));
                    }
                    m_state.value = seed;
                    m_valueSignal(m_state.value);
                    m_state.lastCount = count;
                    m_state.lastSeed = seed;
                    m_state.lastSeedTime = timestamp;
                    if (auto simulation = dynamic_cast<Simulation*>(m_simulation)) {
                        simulation->logDebug("FuturesSignal::update : PUBLISH {}", m_state.value);
                    }
                }
            } else {
                if (m_state.lastCount > 0) {
                    fmt::println("FuturesSignal::update : NO SEED FILE PRESENT AT {}", m_seedfile);
                }            
            }
        }
    }
    else {
        m_state.value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
        m_valueSignal(m_state.value);
    }
}

//-------------------------------------------------------------------------

std::unique_ptr<FuturesSignal> FuturesSignal::fromXML(
    taosim::simulation::ISimulation* simulation, pugi::xml_node node, uint64_t bookId, double X0)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getNonNegativeFloatAttribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (double value = attr.as_double(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };

    auto getNonNegativeUint64Attribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (uint64_t value = attr.as_ullong(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };

    return std::make_unique<FuturesSignal>(FuturesSignalDesc{
        .simulation = simulation,
        .bookId = bookId,
        .seedInterval = getNonNegativeUint64Attribute(node, "seedInterval"),
        .X0 = X0,
        .lambda = node.attribute("lambda").as_double(0.001155),
        .proc = {
            .updatePeriod = node.attribute("updatePeriod").as_ullong(1)
        }
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------