/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "IsPointer.hpp"

//-------------------------------------------------------------------------

namespace taosim::mp
{

template<typename T, typename U>
concept IsTypeOrPointerToType = std::same_as<T, U> || IsPointer<T>;

}  // namespace taosim::mp

//-------------------------------------------------------------------------
