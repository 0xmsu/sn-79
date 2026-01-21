/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/filesystem/utils.hpp>

//-------------------------------------------------------------------------

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

//-------------------------------------------------------------------------

std::vector<fs::path> collectMatchingPaths(
    const fs::path& dir, std::function<bool(const fs::path&)> criterion)
{
    std::vector<fs::path> res;
    for (auto&& entry : fs::directory_iterator(dir)) {
        const auto path = entry.path();
        if (criterion(path)) {
            res.push_back(path);
        }
    }
    return res;
}

//-------------------------------------------------------------------------

std::vector<fs::path> collectPaths(const fs::path& dir)
{
    return collectMatchingPaths(dir, [](auto) { return true; });
}

//-------------------------------------------------------------------------

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------