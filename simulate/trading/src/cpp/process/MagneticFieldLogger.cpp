/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/MagneticFieldLogger.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

MagneticFieldLogger::MagneticFieldLogger(const logging::LoggerBaseDesc& desc)
    : logging::LoggerBase(desc)
{}

//-------------------------------------------------------------------------

void MagneticFieldLogger::log(
    Timestamp timestamp,
    float totalMagnetism,
    std::span<const int32_t> field,
    uint32_t lastPosition)
{
    m_logger->trace(fmt::format(
        "{},{},{},{}", timestamp, totalMagnetism, fmt::join(field, ";"), lastPosition));
    m_logger->flush();
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
