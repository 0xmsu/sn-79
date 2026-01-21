/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <concepts>

//-------------------------------------------------------------------------

namespace taosim::mp
{

//-------------------------------------------------------------------------
// Reference: C++20 The Complete Guide, p. 39. Nicolai M. Josuttis, 2022.

template<typename T>
concept IsPointer = requires (T p) {
    *p;
    p == nullptr;
    {p < p} -> std::convertible_to<bool>;
};

//-------------------------------------------------------------------------

}  // namespace taosim::mp

//-------------------------------------------------------------------------
