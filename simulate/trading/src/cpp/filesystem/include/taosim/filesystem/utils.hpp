/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <filesystem>
#include <functional>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

[[nodiscard]] std::vector<std::filesystem::path> collectMatchingPaths(
    const std::filesystem::path& dir, std::function<bool(const std::filesystem::path&)> criterion);

[[nodiscard]] std::vector<std::filesystem::path> collectPaths(const std::filesystem::path& dir);

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------