/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/Process.hpp>
#include <taosim/process/RNG.hpp>
#include <taosim/simulation/ISimulation.hpp>
#include "common.hpp"

#include <pugixml.hpp>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct FuturesSignalDesc
{
    simulation::ISimulation* simulation;
    uint64_t bookId;
    uint64_t seedInterval;
    double X0;
    double lambda;
    ProcessDesc proc;
};

struct FuturesSignalState
{
    double logReturn{};
    double volumeFactor{2.0};
    uint32_t factorCounter{};
    uint64_t lastCount{};
    double lastSeed{};
    Timestamp lastSeedTime{};
    double value{};
};

//-------------------------------------------------------------------------

class FuturesSignal : public Process
{
public:
    FuturesSignal() noexcept = default;
    FuturesSignal(const FuturesSignalDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }    

    [[nodiscard]] double volumeFactor() noexcept;

    virtual void update(Timestamp timestamp) override;
    virtual double value() const override { return m_state.value; };
    virtual uint64_t count() const override { return m_state.lastCount; };

    [[nodiscard]] static std::unique_ptr<FuturesSignal> fromXML(
        simulation::ISimulation* simulation, pugi::xml_node node, uint64_t bookId, double X0);

private:
    simulation::ISimulation* m_simulation;
    uint64_t m_bookId;
    uint64_t m_seedInterval;
    double m_lambda;
    std::string m_seedfile;
    FuturesSignalState m_state;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
