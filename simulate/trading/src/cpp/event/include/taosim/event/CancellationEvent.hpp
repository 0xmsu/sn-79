/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/Cancellation.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

struct CancellationEvent : public JsonSerializable
{
    Cancellation cancellation;
    Timestamp timestamp;
    taosim::decimal_t price;

    CancellationEvent() noexcept = default;

    CancellationEvent(
        Cancellation cancellation, Timestamp timestamp, taosim::decimal_t price) noexcept
        : cancellation{cancellation}, timestamp{timestamp}, price{price}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;
};

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------
