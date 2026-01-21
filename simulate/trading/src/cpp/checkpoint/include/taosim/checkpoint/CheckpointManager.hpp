/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <filesystem>
#include <regex>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

class SimulationManager;

}  // namespace taosim::simulation

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

struct CheckpointingDesc
{
    simulation::SimulationManager* simuMngr;
    std::filesystem::path runDir;
    size_t intervalInSteps{};
    ssize_t numLastFilesToKeep{};
    bool measureWallClockTime{};
};

//-------------------------------------------------------------------------

class CheckpointManager
{
public:
    explicit CheckpointManager(const CheckpointingDesc& desc);

    [[nodiscard]] auto&& stepCounter(this auto&& self) noexcept { return self.m_stepCounter; }

    void saveCheckpoint();

    static constexpr std::string_view s_storeDirName{"ckpt"};
    static constexpr std::string_view s_dirExtension{".ckptd"};
    static constexpr std::string_view s_fileExtension{".ckpt"};
    static inline const std::regex s_relevantLogFilePattern{R"(((L[23]|fees).*\.log)|(.*\.csv))"};

private:
    void saveCheckpointImpl();
    void saveCheckpointMeasured();
    void cleanup();

    simulation::SimulationManager* m_simuMngr;
    std::filesystem::path m_dir;
    size_t m_intervalInSteps;
    ssize_t m_numLastFilesToKeep;
    bool m_measureWallClockTime;
    ssize_t m_stepCounter{-1};
    std::filesystem::path m_latestCkptDir;

    friend class simulation::SimulationManager;
};

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
