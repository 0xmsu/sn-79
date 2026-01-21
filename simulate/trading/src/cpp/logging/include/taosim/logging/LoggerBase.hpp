/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <filesystem>
#include <memory>

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

struct LoggerBaseDesc
{
    std::string name;
    std::filesystem::path filepath;
    std::string header;
};

//-------------------------------------------------------------------------

class LoggerBase
{
public:
    explicit LoggerBase(const LoggerBaseDesc& desc);

    [[nodiscard]] const std::filesystem::path& filepath() const noexcept { return m_filepath; }

protected:
    std::unique_ptr<spdlog::logger> m_logger;
    std::filesystem::path m_filepath;
};

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------