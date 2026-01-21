/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/replay/ReplayManager.hpp>

#include <taosim/filesystem/utils.hpp>
#include <taosim/replay/helpers.hpp>

//-------------------------------------------------------------------------

namespace taosim::replay
{

//-------------------------------------------------------------------------

ReplayManager::ReplayManager(const ReplayDesc& desc, ReplayManager::ValidatorFn validator)
{
    validator(desc);

    m_desc = desc;

    fmt::println("{}", m_desc);

    populateInitialBalancesPaths();
    populateRuntimePathGroups();
}

//-------------------------------------------------------------------------

void ReplayManager::populateInitialBalancesPaths()
{
    static const std::regex pat{R"(^Replay-Balances-(\d+)-(\d+)\.json$)"};

    auto paths = filesystem::collectMatchingPaths(
        m_desc.dir,
        [&](const fs::path& p) {
            return fs::is_regular_file(p) && std::regex_match(p.filename().string(), pat);
        })
        | ranges::actions::sort([&](auto&& lhs, auto&& rhs) {
            const auto lhsStr = lhs.filename().string();
            std::smatch matchLhs;
            std::regex_search(lhsStr, matchLhs, pat);
            const auto rhsStr = rhs.filename().string();
            std::smatch matchRhs;
            std::regex_search(rhsStr, matchRhs, pat);
            return std::stoi(matchLhs[1]) < std::stoi(matchRhs[1]);
        });

    m_bookCountTotal = [&] {
        const auto lastFilenameStr = paths.back().filename().string();
        if (std::smatch match; std::regex_match(lastFilenameStr, match, pat)) {
            return std::stoul(match[2]) + 1;
        }
        throw helpers::ReplayError{};
    }();

    m_initialBalancesPaths = std::move(paths);
}

//-------------------------------------------------------------------------

void ReplayManager::populateRuntimePathGroups()
{
    if (m_bookCountTotal == 0uz) {
        populateInitialBalancesPaths();
    }

    auto parseCommon = [&](auto&& path, const std::regex& pat, size_t idx) {
        const auto filenameStr = path.filename().string();
        if (std::smatch match; std::regex_match(filenameStr, match, pat)) {
            return std::stoul(match[idx]);
        }
        throw helpers::ReplayError{};
    };
    auto parseBookId = [&](auto&& path, const std::regex& pat) {
        return parseCommon(path, pat, 1);
    };
    auto parseBeginTime = [&](auto&& path, const std::regex& pat) {
        return parseCommon(path, pat, 2);
    };

    auto getMatchingPathsSorted = [&](const std::regex& pat) {
        const auto paths = filesystem::collectMatchingPaths(
            m_desc.dir,
            [&](const fs::path& p) {
                return fs::is_regular_file(p) && std::regex_match(p.filename().string(), pat);
            });
        std::vector<std::vector<fs::path>> res(m_bookCountTotal);
        for (auto&& path : paths) {
            const auto bookIdCanon = parseBookId(path, pat);
            res.at(bookIdCanon).push_back(path);
        }
        for (auto&& paths : res) {
            ranges::sort(
                paths,
                [&](auto&& lhs, auto&& rhs) {
                    return parseBeginTime(lhs, pat) < parseBeginTime(rhs, pat);
                });
        }
        return res;
    };

    const auto instrLogPaths = getMatchingPathsSorted(ReplayRuntimePathGroup::s_patInstr);
    const auto L2LogPaths = getMatchingPathsSorted(ReplayRuntimePathGroup::s_patL2);

    auto logPathsZipped = views::zip(instrLogPaths, L2LogPaths);

    std::vector<std::vector<ReplayRuntimePathGroup>> res(m_bookCountTotal);
    for (auto&& [blockIdx, item]: views::enumerate(logPathsZipped)) {
        for (auto&& pathTuple : views::zip(item.first, item.second)) {
            res.at(blockIdx).push_back(ReplayRuntimePathGroup{
                .instr = pathTuple.first,
                .L2 = pathTuple.second
            });
        }
    }
    m_runtimePathGroups = std::move(res);
}

//-------------------------------------------------------------------------

}  // namespace taosim::replay

//-------------------------------------------------------------------------