/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::replay
{

struct ReplayDesc
{
    fs::path dir;
    std::optional<BookId> bookId;
    std::set<std::string> replacedAgents;
    bool adjustLimitPrices{};
    bool removeExistingDir{};
};

}  // namespace taosim::replay

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::replay::ReplayDesc>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::replay::ReplayDesc& val, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "ReplayDesc{{\n"
            "    dir = {}\n"
            "    bookId = {}\n"
            "    replacedAgents = {}\n"
            "    adjustLimitPrices = {}\n"
            "    removeExistingDir = {}\n"
            "}}",
            val.dir.c_str(),
            fmt::format("{}", val.bookId ? std::to_string(*val.bookId) : "nullopt"),
            val.replacedAgents,
            val.adjustLimitPrices,
            val.removeExistingDir);
    }
};

//-------------------------------------------------------------------------
