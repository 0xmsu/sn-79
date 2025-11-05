/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "FuturesSignal.hpp"

#include "Simulation.hpp"

#include <cmath>
#include <source_location>

//-------------------------------------------------------------------------

FuturesSignal::FuturesSignal(
    taosim::simulation::ISimulation* simulation,
    uint64_t bookId,
    uint64_t seedInterval,
    double X0,
    Timestamp updatePeriod) noexcept
    : m_simulation{simulation},
      m_bookId{bookId},
      m_seedInterval{seedInterval},
      m_value{X0}
{
    m_updatePeriod = updatePeriod;
    m_value = std::round(m_X0);
    m_seedfile = (simulation->logDir() / "external_seed_sampled.csv").generic_string();
}

//-------------------------------------------------------------------------

void FuturesSignal::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        if (timestamp - m_last_seed_time >= m_seedInterval) {
            if ( fs::exists( m_seedfile ) ) {
                int count = m_last_count;
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
                        if (m_last_count > 0) {
                            fmt::println("FuturesSignal::update : FAILED TO GET SEED FROM FILE - NO DATA ({} LINES READ)", lines.size());
                        }                    
                    }
                } catch (const std::exception &exc) {
                    fmt::println("FuturesSignal::update : ERROR GETTING SEED FROM FILE - {}", exc.what());
                }
                if (count != m_last_count) {
                    m_value = seed;
                    m_valueSignal(m_value);
                    m_last_count = count;
                    m_last_seed = seed;
                    m_last_seed_time = timestamp;
                    if (auto simulation = dynamic_cast<Simulation*>(m_simulation)) {
                        simulation->logDebug("FuturesSignal::update : PUBLISH {}", m_value);
                    }
                }
            } else {
                if (m_last_count > 0) {
                    fmt::println("FuturesSignal::update : NO SEED FILE PRESENT AT {}", m_seedfile);
                }            
            }
        }
    }
    else {
        m_value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
        m_valueSignal(m_value);
    }
}

//-------------------------------------------------------------------------

void FuturesSignal::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("name", rapidjson::Value{"external", allocator}, allocator);
        json.AddMember("bookId", rapidjson::Value{m_bookId}, allocator);
        json.AddMember("seedInterval", rapidjson::Value{m_seedInterval}, allocator);
        json.AddMember("X0", rapidjson::Value{m_X0}, allocator);
        json.AddMember("value", rapidjson::Value{m_value}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
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

    return std::make_unique<FuturesSignal>(
        simulation,
        bookId,
        getNonNegativeUint64Attribute(node, "seedInterval"),
        X0,
        node.attribute("updatePeriod").as_ullong(1));
}

//-------------------------------------------------------------------------

std::unique_ptr<FuturesSignal> FuturesSignal::fromCheckpoint(
    taosim::simulation::ISimulation* simulation, const rapidjson::Value& json, double X0)
{
    auto fp = std::make_unique<FuturesSignal>(
        simulation,
        json["bookId"].GetUint64(),
        json["seedInterval"].GetUint64(),
        X0,
        1);
    fp->m_value = json["value"].GetDouble();
    return fp;
}

//-------------------------------------------------------------------------
