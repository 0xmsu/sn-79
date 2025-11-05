/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "GBM.hpp"

//-------------------------------------------------------------------------

GBM::GBM(double X0, double mu, double sigma, double dt, Timestamp updatePeriod) noexcept
    : m_X0{X0},
      m_mu{mu},
      m_sigma{sigma},
      m_dt{dt},
      m_gaussian{0.0, std::sqrt(dt)},
      m_value{X0}
{
    m_updatePeriod = updatePeriod;
}

//-------------------------------------------------------------------------

GBM::GBM(double X0, double mu, double sigma, double dt, uint64_t seed, Timestamp updatePeriod) noexcept
    : GBM{X0, mu, sigma, dt, updatePeriod}
{
    m_updatePeriod = updatePeriod;
    m_rng = RNG{seed};
}

//-------------------------------------------------------------------------

void GBM::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        m_t += m_dt;
        m_W += m_gaussian(m_rng);
        m_value = m_X0 * std::exp((m_mu - 0.5 * m_sigma * m_sigma) * m_t + m_sigma * m_W);
    }
    else {
        m_value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
    }
    m_valueSignal(m_value);
}

//-------------------------------------------------------------------------

void GBM::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("name", rapidjson::Value{"GBM", allocator}, allocator);
        m_rng.checkpointSerialize(json, "rng");
        json.AddMember("X0", rapidjson::Value{m_X0}, allocator);
        json.AddMember("mu", rapidjson::Value{m_mu}, allocator);
        json.AddMember("sigma", rapidjson::Value{m_sigma}, allocator);
        json.AddMember("dt", rapidjson::Value{m_dt}, allocator);
        json.AddMember("t", rapidjson::Value{m_t}, allocator);
        json.AddMember("W", rapidjson::Value{m_W}, allocator);
        json.AddMember("value", rapidjson::Value{m_value}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

std::unique_ptr<GBM> GBM::fromXML(pugi::xml_node node, uint64_t seedShift)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getNonNegativeAttribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (double value = attr.as_double(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };

    const uint64_t seed = [&] {
        pugi::xml_attribute attr;
        if (attr = node.attribute("seed"); attr.empty()) {
            throw std::invalid_argument(fmt::format(
                "{}: Missing required attribute '{}'", ctx, "seed"));
        }
        return attr.as_ullong();
    }();

    return std::make_unique<GBM>(
        getNonNegativeAttribute(node, "X0"),
        getNonNegativeAttribute(node, "mu"),
        getNonNegativeAttribute(node, "sigma"),
        getNonNegativeAttribute(node, "dt"),
        seed + seedShift,
        node.attribute("updatePeriod").as_ullong(1));
}

//-------------------------------------------------------------------------

std::unique_ptr<GBM> GBM::fromCheckpoint(const rapidjson::Value& json)
{
    auto gbm = std::make_unique<GBM>(
        json["X0"].GetDouble(),
        json["mu"].GetDouble(),
        json["sigma"].GetDouble(),
        json["dt"].GetDouble(),
        1);
    gbm->m_t = json["t"].GetDouble();
    gbm->m_W = json["W"].GetDouble();
    gbm->m_value = json["value"].GetDouble();
    gbm->m_rng = RNG::fromCheckpoint(json["rng"]);
    return gbm;
}

//-------------------------------------------------------------------------
