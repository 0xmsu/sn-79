/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/decimal.hpp>
#include "JsonSerializable.hpp"
#include "Order.hpp"

#include <deque>
#include <list>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class OrderContainer;

//-------------------------------------------------------------------------

class TickContainer
    : public std::list<LimitOrder::Ptr>,
      public JsonSerializable
{
public:
    using BaseType = std::list<value_type>;

    using BaseType::BaseType;

    TickContainer(OrderContainer* orderContainer, taosim::decimal_t price) noexcept;

    [[nodiscard]] auto&& orderContainer(this auto&& self) noexcept { return self.m_orderContainer; }
    [[nodiscard]] auto&& price(this auto&& self) noexcept { return self.m_price; }
    [[nodiscard]] auto&& volume(this auto&& self) noexcept { return self.m_volume; }

    void updateVolume(taosim::decimal_t deltaVolume) noexcept;

    bool operator<(const TickContainer& rhs) const noexcept { return m_price < rhs.price(); }
    bool operator<(taosim::decimal_t price) const noexcept { return m_price < price; }

    void push_back(const value_type& elem);
    void pop_front();

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

private:
    OrderContainer* m_orderContainer;
    taosim::decimal_t m_price;
    taosim::decimal_t m_volume{};
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
