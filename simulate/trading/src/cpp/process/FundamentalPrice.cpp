/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/FundamentalPrice.hpp>

#include <Simulation.hpp>

#include <cmath>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

FundamentalPrice::FundamentalPrice(const FundamentalPriceDesc& desc) noexcept
    : m_simulation{desc.simulation},
      m_rng{&dynamic_cast<Simulation*>(desc.simulation)->rng()},
      m_bookId{desc.bookId},
      m_seedInterval{desc.seedInterval},
      m_mu{desc.mu},
      m_sigma{desc.sigma},
      m_dt{desc.dt},
      m_gaussian{0.0, std::sqrt(desc.dt)},
      m_X0{desc.X0},
      m_poisson{desc.lambda},
      m_jump{desc.muJump, desc.sigmaJump},
      m_hurst{desc.hurst},
      m_epsilon{desc.epsilon}
{
    m_updatePeriod = desc.proc.updatePeriod;
    m_state.value = m_X0;
 
    const auto sim = dynamic_cast<Simulation*>(m_simulation);
    Timestamp N = sim->duration() / m_updatePeriod;
    const double dtH = std::pow(N, -m_hurst);
    m_state.L = Eigen::MatrixXd::Zero(N + 2, N + 2);
    m_state.X = Eigen::VectorXd::Zero(N + 2);
    m_state.V.resize(N + 2);
    // TODO seed
    m_fractionalGaussian = std::normal_distribution<double>{0.0, dtH};

    for (int i = 0; i < 2; i++) {
        m_state.V(i) = m_epsilon * m_fractionalGaussian(*m_rng);
    }

    m_state.L(0,0) = 1.0;
    m_state.X(0) = m_state.V(0);

    m_state.L(1,0) = gamma_fn(1, m_hurst);
    m_state.L(1,1) = std::sqrt(1.0 - std::pow(m_state.L(1,0), 2));
    m_state.X(1) = m_state.L.row(1).head(2).dot(m_state.V.head(2));

    m_seedfile = (sim->logDir() / "fundamental_seed.csv").generic_string();
}

//-------------------------------------------------------------------------

void FundamentalPrice::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        if (timestamp - m_state.lastSeedTime >= m_seedInterval) {
            int count = m_state.lastCount;
            uint64_t seed = 0;
            if ( fs::exists( m_seedfile ) ) {
                try {
                    std::vector<std::string> lines = taosim::util::getLastLines(m_seedfile, 2);
                    if (lines.size() >= 2) {
                        std::vector<std::string> line = taosim::util::split(lines[lines.size() - 2], ',');
                        if (line.size()== 2) {
                            count = std::stoi(line[0]);
                            seed = static_cast<uint64_t>(round(std::stof(line[1])*100)) + m_bookId*10;
                        } else {
                            fmt::println("FundamentalPrice::update : FAILED TO GET SEED FROM LINE - {}", lines[lines.size() - 2]);
                        }
                    } else {
                        fmt::println("FundamentalPrice::update : FAILED TO GET SEED FROM FILE - NO DATA ({} LINES READ)", lines.size());
                    }
                } catch (const std::exception& exc) {
                    fmt::println("FundamentalPrice::update : ERROR GETTING SEED FROM FILE - {}", exc.what());
                }
                if (count == m_state.lastCount) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> distr(-50, 50);
                    seed = m_state.lastSeed + distr(gen);
                    fmt::println("WARNING : Fundamental price seed not updated - using random seed.  Last Count {} | Count {} | Last Seed {} | Seed {}", m_state.lastCount, count, seed, m_state.lastSeed);
                }
            } else {
                fmt::println("FundamentalPrice::update : NO SEED FILE PRESENT AT {}.  Using random seed.", m_seedfile);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distr(10800000,11200000);
                seed = distr(gen);
            }
            m_rng->seed(seed); 
            m_state.lastCount = count;
            m_state.lastSeed = seed;
            m_state.lastSeedTime = timestamp; 
            m_state.t += m_dt;
            // Jump part
            m_state.dJ += m_poisson(*m_rng) * m_jump(*m_rng);
            //fBM
            int64_t step = timestamp/m_updatePeriod;
            cholesky_step(step);
            m_state.BH += m_state.X(step);
            const double fBM_comp = 
                m_epsilon * m_state.BH - (0.5 * m_epsilon * m_epsilon * std::pow(m_state.t, 2 * m_hurst));
            // BM 
            m_state.W += m_gaussian(*m_rng);
            // pricing
            m_state.value = m_X0 * std::exp((m_mu - 0.5 * m_sigma * m_sigma) * m_state.t + m_sigma * m_state.W + fBM_comp + m_state.dJ);
        }
    }
    else {
        m_state.value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
    }
    m_valueSignal(m_state.value);
}

//-------------------------------------------------------------------------

void FundamentalPrice::cholesky_step(int64_t i)
{
    m_state.L(i, 0) = gamma_fn(i, m_hurst);
    m_state.V(i + 1) = m_fractionalGaussian(*m_rng);

    for (int j = 1; j < i; j++)
    {
        double dot_val = m_state.L.row(i).head(j).dot(m_state.L.row(j).head(j));
        m_state.L(i, j) = (1.0 / m_state.L(j, j)) * (gamma_fn(i - j, m_hurst) - dot_val);
    }

    double sumsq = m_state.L.row(i).head(i).squaredNorm();
    m_state.L(i, i) = std::sqrt(1.0 - sumsq);

    m_state.X(i) = m_state.L.row(i).head(i + 1).dot(m_state.V.head(i + 1));
}

//-------------------------------------------------------------------------

double FundamentalPrice::gamma_fn(int64_t k, double H) const
{
    return 0.5 * (std::pow(std::abs(k - 1), 2.0 * H)
                - 2.0 * std::pow(std::abs(k), 2.0 * H)
                + std::pow(std::abs(k + 1), 2.0 * H));
}

//-------------------------------------------------------------------------

std::unique_ptr<FundamentalPrice> FundamentalPrice::fromXML(
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

    const auto updatePeriod = node.attribute("updatePeriod").as_ullong(1);
    const auto sim = dynamic_cast<Simulation*>(simulation);
    const float dt = (float) updatePeriod / sim->duration();

    auto getNonNegativeUint64Attribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (uint64_t value = attr.as_ullong(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };
    const double hurst = node.attribute("Hurst").as_double(0.5); 
    const double epsilon = node.attribute("epsilon").as_double(0.0);
    return std::make_unique<FundamentalPrice>(FundamentalPriceDesc{
        .simulation = simulation,
        .bookId = bookId,
        .seedInterval = getNonNegativeUint64Attribute(node, "seedInterval"),
        .X0 = X0,      
        .mu = getNonNegativeFloatAttribute(node, "mu"),
        .sigma = getNonNegativeFloatAttribute(node, "sigma"),
        .dt = dt,
        .lambda = getNonNegativeFloatAttribute(node, "lambda"),
        .muJump = getNonNegativeFloatAttribute(node, "muJump"),
        .sigmaJump = getNonNegativeFloatAttribute(node, "sigmaJump"),
        .hurst = hurst, 
        .epsilon = epsilon,
        .proc = {
            .updatePeriod = updatePeriod
        }
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------