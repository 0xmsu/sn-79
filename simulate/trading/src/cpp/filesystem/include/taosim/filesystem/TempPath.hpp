/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>

#include <filesystem>
#include <stdexcept>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

class TempPath
{
public:
    explicit TempPath(const std::filesystem::path& path)
    {
        if (path.empty()) {
            throw std::invalid_argument{fmt::format(
                "{}: 'path' must be non-empty",
                std::source_location::current().function_name()
            )};
        }
        m_path = std::filesystem::temp_directory_path() / path;
    }

    ~TempPath() noexcept
    {
        std::filesystem::remove(m_path);
    }

    operator const std::filesystem::path&() const { return m_path; }
    operator const char*() const { return m_path.c_str(); }

private:
    std::filesystem::path m_path;
};

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------
