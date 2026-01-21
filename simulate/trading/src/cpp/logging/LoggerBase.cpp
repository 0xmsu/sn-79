/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/logging/LoggerBase.hpp>

//-------------------------------------------------------------------------

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

LoggerBase::LoggerBase(const LoggerBaseDesc& desc)
    : m_filepath{desc.filepath}
{
    const bool logFileExisted = fs::exists(m_filepath);

    m_logger = std::make_unique<spdlog::logger>(
        desc.name, std::make_unique<spdlog::sinks::basic_file_sink_st>(m_filepath));
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("%v");
    
    if (logFileExisted) return;

    m_logger->trace(desc.header);
    m_logger->flush();
}

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------