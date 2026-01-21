/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/logging/LoggerBase.hpp>
#include <Timestamp.hpp>

#include <cstdint>
#include <span>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

class MagneticFieldLogger : public logging::LoggerBase
{
public:
    explicit MagneticFieldLogger(const logging::LoggerBaseDesc& desc);

    void log(
        Timestamp timestamp,
        float totalMagnetism,
        std::span<const int32_t> field,
        uint32_t lastPosition = 0);

    static constexpr std::string_view s_header = "time,total,field,lastPosition"; 
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
