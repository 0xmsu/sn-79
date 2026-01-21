/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/TimeConfig.hpp>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

TimeConfig::TimeConfig(Timestamp start, Timestamp duration, Timestamp step, Timescale scale)
    : start{start}, duration{duration}, step{step}, scale{scale}
{}

//-------------------------------------------------------------------------

TimeConfig TimeConfig::fromXML(pugi::xml_node node)
{
    auto timescaleFallback = [] {
        static constexpr auto fallback = Timescale::s;
        fmt::println("Unknown or missing attribute 'timescale', falling back to '{}'", fallback);
        return std::make_optional(fallback);
    };

    return TimeConfig{
        node.attribute("start").as_ullong(),
        node.attribute("duration").as_ullong(),
        node.attribute("step").as_ullong(),
        magic_enum::enum_cast<Timescale>(
            node.attribute("timescale").as_string()).or_else(timescaleFallback).value()};
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
