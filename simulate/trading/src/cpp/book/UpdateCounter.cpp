/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/UpdateCounter.hpp>

#include <fmt/format.h>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

UpdateCounter::UpdateCounter(Timestamp period) noexcept
{
    m_internalPeriod = period == 0 ? period : period - 1;
}

//-------------------------------------------------------------------------

UpdateCounter UpdateCounter::fromXML(pugi::xml_node node)
{
    return UpdateCounter{node.attribute("updatePeriod").as_ullong()};
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
