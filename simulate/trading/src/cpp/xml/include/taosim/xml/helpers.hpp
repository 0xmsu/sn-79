/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

//-------------------------------------------------------------------------

namespace taosim::xml
{

void setAttribute(pugi::xml_node node, const char* name, const auto& value)
{
    if (auto attr = node.attribute(name)) {
        attr.set_value(value);
    } else {
        node.append_attribute(name) = value;
    }
}

}  // namespace taosim::xml

//-------------------------------------------------------------------------