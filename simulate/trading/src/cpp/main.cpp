/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/replay/ReplayDesc.hpp"
#include "taosim/replay/helpers.hpp"
#include "taosim/simulation/SimulationManager.hpp"
#include "common.hpp"

#include <CLI/CLI.hpp>
#ifdef OVERRIDE_NEW_DELETE
#include <mimalloc-new-delete.h>
#endif
#include <pybind11/embed.h>

namespace py = pybind11;

//-------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    CLI::App app{"ExchangeSimulator v2.0"};

    py::scoped_interpreter guard{};

    CLI::Option_group* initGroup = app.add_option_group("Init");

    fs::path config;
    initGroup->add_option("-f,--config-file", config, "Simulation config file")
        ->check(CLI::ExistingFile);

    fs::path checkpoint;
    initGroup->add_option("-c,--checkpoint-file", checkpoint, "Checkpoint file")
        ->check(CLI::ExistingFile);

    taosim::replay::ReplayDesc replayDesc;

    auto optReplayDir = initGroup->add_option(
        "-r,--replay-dir", replayDesc.dir, "Log directory to use in a replay context")
        ->check(CLI::ExistingDirectory)
        ->transform(&taosim::replay::helpers::cleanReplayPath);

    app.add_option("--book-id", replayDesc.bookId, "Book to replay")
        ->needs(optReplayDir);

    app.add_option(
        "--replaced-agents",
        replayDesc.replacedAgents,
        "Comma-separated list of agent base names which to replace during replay")
        ->delimiter(',')
        ->needs(optReplayDir);

    app.add_flag(
        "--adjust-limit-prices",
        replayDesc.adjustLimitPrices,
        "Adjust limit prices of passive agents using historical mid price data")
        ->needs(optReplayDir);

    app.add_flag(
        "--rm,--remove-existing-dir",
        replayDesc.removeExistingDir,
        "Remove potentially existing replay directory (DOESN'T DO ANYTHING CURRENTLY)")
        ->needs(optReplayDir);

    initGroup->require_option(1);

    CLI11_PARSE(app, argc, argv);

    fmt::println("{}", app.get_description());

    if (!checkpoint.empty()) {
        throw std::runtime_error{"Loading from checkpoint currently unsupported!"};
    }
    if (!replayDesc.dir.empty()) {
        auto mngr = taosim::simulation::SimulationManager::fromReplay(replayDesc);
        if (replayDesc.bookId) {
            mngr->runReplay();
        } else {
            mngr->runReplayAdvanced();
        }
    }
    else {
        auto mngr = taosim::simulation::SimulationManager::fromConfig(config);
        mngr->runSimulations();
    }

    fmt::println(" - all simulations finished, exiting");

    return 0;
}

//-------------------------------------------------------------------------
