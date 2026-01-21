/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookProcessLogger.hpp>
#include <taosim/book/UpdateCounter.hpp>
#include <taosim/process/Process.hpp>
#include <taosim/process/ProcessFactory.hpp>
#include <taosim/simulation/SimulationSignals.hpp>
#include <CheckpointSerializable.hpp>
#include "common.hpp"

#include <pugixml.hpp>

//-------------------------------------------------------------------------

class Simulation;

namespace taosim::exchange
{

class ExchangeConfig;

}  // namespace taosim::exchange

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class BookProcessManager
{
public:
    using ProcessContainer = std::map<std::string, std::vector<std::unique_ptr<taosim::process::Process>>>;
    using LoggerContainer = std::map<std::string, std::unique_ptr<BookProcessLogger>>;
    using UpdateCounterContainer = std::map<std::string, UpdateCounter>;

    BookProcessManager() noexcept = default;
    BookProcessManager(
        ProcessContainer container,
        LoggerContainer loggers,
        std::unique_ptr<taosim::process::ProcessFactory> processFactory,
        decltype(taosim::simulation::SimulationSignals::time)& timeSignal);

    [[nodiscard]] auto&& container(this auto&& self) noexcept { return self.m_container; }
    [[nodiscard]] auto&& updateCounters(this auto&& self) noexcept { return self.m_updateCounters; }

    [[nodiscard]] auto&& operator[](this auto&& self, const std::string& name)
    {
        return self.m_container[name];
    }

    [[nodiscard]] auto&& at(this auto&& self, const std::string& name)
    {
        return self.m_container.at(name);
    }

    void updateProcesses(Timespan timespan);

    [[nodiscard]] static std::unique_ptr<BookProcessManager> fromXML(
        pugi::xml_node node, Simulation* simulation, taosim::exchange::ExchangeConfig* exchangeConfig);

private:
    ProcessContainer m_container;
    LoggerContainer m_loggers;
    std::unique_ptr<taosim::process::ProcessFactory> m_processFactory;
    bs2::scoped_connection m_feed;
    UpdateCounterContainer m_updateCounters;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------