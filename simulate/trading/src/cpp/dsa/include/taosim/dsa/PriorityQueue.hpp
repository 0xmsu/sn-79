/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <queue>

//-------------------------------------------------------------------------

namespace taosim::dsa
{

template<
    typename T,
    typename Container = std::vector<T>,
    typename Compare = std::less<typename Container::value_type>>
class PriorityQueue : public std::priority_queue<T, Container, Compare>
{
public:
    using BaseType = std::priority_queue<T, Container, Compare>;
    using CompareType = Compare;

    using BaseType::BaseType;

    [[nodiscard]] const Container& underlying() const noexcept { return BaseType::c; }

    void clear() noexcept { BaseType::c.clear(); }
};

}  // taosim::dsa

//-------------------------------------------------------------------------