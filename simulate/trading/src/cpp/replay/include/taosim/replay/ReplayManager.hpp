/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/replay/ReplayDesc.hpp>

//-------------------------------------------------------------------------

namespace taosim::replay
{

//-------------------------------------------------------------------------

struct ReplayRuntimePathGroup
{
    fs::path instr;
    fs::path L2;

    static inline const std::regex s_patInstr{R"(^Replay-(\d+)\.(\d{8})-(\d{8})\.log$)"};
    static inline const std::regex s_patL2{R"(^L2-(\d+)\.(\d{8})-(\d{8})\.log$)"};
};

//-------------------------------------------------------------------------

class ReplayManager
{
public:
    using ValidatorFn = std::function<void(const ReplayDesc& desc)>;

    ReplayManager(const ReplayDesc& desc, ValidatorFn validator = [](auto&&...){});

    [[nodiscard]] auto&& desc(this auto&& self) noexcept { return self.m_desc; }

    [[nodiscard]] std::span<const fs::path> initialBalancesPaths() const noexcept
    {
        return m_initialBalancesPaths;
    }

    [[nodiscard]] std::span<const std::vector<ReplayRuntimePathGroup>>
        runtimePathGroups() const noexcept
    {
        return m_runtimePathGroups;
    }

private:
    void populateInitialBalancesPaths();
    void populateRuntimePathGroups();

    ReplayDesc m_desc;
    size_t m_bookCountTotal{};
    std::vector<fs::path> m_initialBalancesPaths;
    std::vector<std::vector<ReplayRuntimePathGroup>> m_runtimePathGroups;
};

//-------------------------------------------------------------------------

}  // namespace taosim::replay

//-------------------------------------------------------------------------
