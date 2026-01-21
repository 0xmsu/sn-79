/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/CancellationEvent.hpp>
#include <taosim/event/EventRecord.hpp>
#include <taosim/event/OrderEvent.hpp>
#include <taosim/event/TradeEvent.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

using L3Record = EventRecord<OrderEvent, TradeEvent, CancellationEvent>;

//-------------------------------------------------------------------------

class L3RecordContainer : public JsonSerializable
{
public:
    L3RecordContainer() noexcept = default;
    explicit L3RecordContainer(uint32_t bookCount) noexcept : m_underlying(bookCount) {}

    auto&& at(this auto&& self, BookId bookId) { return self.m_underlying.at(bookId); }

    [[nodiscard]] auto&& underlying(this auto&& self) noexcept { return self.m_underlying; }

    void clear() noexcept;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

private:
    std::vector<L3Record> m_underlying;
};

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------