/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/Message.hpp>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

Message::Ptr canonize(Message::Ptr msg, uint32_t blockIdx, uint32_t blockDim);

struct DecanonizeResult
{
    Message::Ptr msg;
    std::optional<uint32_t> blockIdx;
};

DecanonizeResult decanonize(Message::Ptr msg, uint32_t blockDim);

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
