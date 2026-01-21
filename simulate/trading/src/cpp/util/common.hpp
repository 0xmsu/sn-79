/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Timestamp.hpp"
#include <taosim/decimal/decimal.hpp>

#include <boost/signals2.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <magic_enum/magic_enum.hpp>
#include <pugixml.hpp>
#include <range/v3/all.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include <tuple>

//-------------------------------------------------------------------------

namespace fs = std::filesystem;

namespace bs2 = boost::signals2;
namespace views = ranges::views;

using namespace taosim::literals;

//-------------------------------------------------------------------------

using OrderID = uint32_t;
using AgentId = int32_t;
using LocalAgentId = std::string;
using BookId = uint32_t;

struct Timespan
{
    Timestamp begin, end;
};

template<typename SlotType>
requires requires { typename std::function<SlotType>; }
using UnsyncSignal = bs2::signal_type<SlotType, bs2::keywords::mutex_type<bs2::dummy_mutex>>::type;

//-------------------------------------------------------------------------
