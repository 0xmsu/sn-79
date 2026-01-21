/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/CheckpointManager.hpp>

#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/SimulatorState.hpp>
#include <taosim/checkpoint/serialization/SimulatorState.hpp>
#include <taosim/checkpoint/helpers.hpp>

#include <fmt/chrono.h>

#include <fstream>
#include <latch>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

CheckpointManager::CheckpointManager(const CheckpointingDesc& desc)
    : m_simuMngr{desc.simuMngr},
      m_intervalInSteps{desc.intervalInSteps},
      m_numLastFilesToKeep{desc.numLastFilesToKeep},
      m_measureWallClockTime{desc.measureWallClockTime}
{
    if (desc.runDir.empty()) {
        throw CheckpointError{"'runDir' must be non-empty"};
    }
    if (m_intervalInSteps == 0) {
        throw CheckpointError{"'intervalInSteps' must be non-zero"};
    }

    m_dir = desc.runDir / s_storeDirName;

    m_simuMngr->stepSignal().connect([this] { saveCheckpoint(); });
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpoint()
{
    if (m_simuMngr->warmingUp()) return;
    if (++m_stepCounter == 0 || m_stepCounter % m_intervalInSteps != 0) return;

    fmt::println("Saving checkpoint...");

    try {
        if (m_measureWallClockTime) {
            saveCheckpointMeasured();
        } else {
            saveCheckpointImpl();
        }
        fmt::println("Checkpoint saved successfully.");
    }
    catch (const std::exception& e) {
        fmt::println("Error saving checkpoint at {}", e.what());
    }
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointImpl()
{
    const auto simuTime = m_simuMngr->simulations().front()->currentTimestamp();
    
    m_latestCkptDir = m_dir / fmt::format("{}{}", simuTime, s_dirExtension);
    std::filesystem::create_directories(m_latestCkptDir);

    // Common.
    taosim::serialization::BinaryStream stream;
    msgpack::packer packer{stream};
    taosim::checkpoint::serialization::packCommon(packer, m_simuMngr);

    const auto commonCkptFile = m_latestCkptDir / fmt::format("common{}", s_fileExtension);
    std::ofstream ofs{commonCkptFile, std::ios::binary};
    ofs.write(stream.data(), stream.size());

    // Blocks.
    std::latch latch{m_simuMngr->blockInfo().count};
    for (auto&& simulation : m_simuMngr->simulations()) {
        boost::asio::post(
            *m_simuMngr->threadPool(),
            [&] {
                taosim::serialization::BinaryStream stream;
                msgpack::packer packer{stream};
                taosim::checkpoint::serialization::packBlock(packer, *simulation);

                const auto blockCkptFile =
                    m_latestCkptDir / fmt::format("{}{}", simulation->blockIdx(), s_fileExtension);
                std::ofstream ofs{blockCkptFile, std::ios::binary};
                ofs.write(stream.data(), stream.size());

                latch.count_down();
            });
    }
    latch.wait();

    cleanup();
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointMeasured()
{
    using namespace std::chrono;

    auto& measurements = m_simuMngr->measurements();

    measurements.t0ckptSave = std::make_optional(high_resolution_clock::now());

    saveCheckpointImpl();

    measurements.t1ckptSave = std::make_optional(high_resolution_clock::now());

    fmt::println(
        "Took {:.4f}s",
        duration<double>(*measurements.t1ckptSave - *measurements.t0ckptSave).count());
}

//-------------------------------------------------------------------------

void CheckpointManager::cleanup()
{
    const auto dirs = ckptDirsSortedByWriteTime(m_dir);

    auto dirsToRemoveView = dirs
        | views::filter([&](auto&& f) {
            static const std::regex pattern{fmt::format("^\\d+\\{}$", s_dirExtension)};
            const auto name = f.filename();
            return std::regex_match(name.string(), pattern)
                && name != m_latestCkptDir.filename();
        })
        | views::take(std::max(0z, std::ssize(dirs) - m_numLastFilesToKeep));

    for (auto&& dir : dirsToRemoveView) {
        fs::remove_all(dir);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
