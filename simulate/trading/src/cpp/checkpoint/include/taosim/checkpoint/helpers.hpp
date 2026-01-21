/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

class SimulationManager;

}  // namespace taosim::simulation

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

[[nodiscard]] CheckpointToken postProcessToken(const CheckpointToken& token);

[[nodiscard]] std::filesystem::path runDirFromToken(const CheckpointToken& token);

using PathFactory = std::function<std::filesystem::path(const std::filesystem::path&)>;

[[nodiscard]] std::filesystem::path runDirLatest(const std::filesystem::path& baseDir);

inline static const std::map<CheckpointToken, PathFactory> s_tokenToRunDirFactory{
    {"latest", &runDirLatest}
};

[[nodiscard]] std::filesystem::path ckptDirLatest(const std::filesystem::path& runDir);

inline static const std::map<CheckpointToken, PathFactory> s_tokenToCkptDirFactory{
    {"latest", &ckptDirLatest}
};

[[nodiscard]] std::filesystem::path ckptDirFromToken(const CheckpointToken& token);

[[nodiscard]] std::vector<std::filesystem::path> ckptDirsSortedByWriteTime(
    const std::filesystem::path& path);

void setupUsingCkptData(
    taosim::simulation::SimulationManager* simuMngr,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles);

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
