/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <set>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

using CheckpointToken = std::string;

inline const std::set<CheckpointToken> s_specialTokens{
    "latest"
};

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
