/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.hpp"
#include "JsonSerializable.hpp"
#include <taosim/serialization/L3Serializable.hpp>

//-------------------------------------------------------------------------

struct PlaceOrderMarketPayload;
struct PlaceOrderLimitPayload;

//-------------------------------------------------------------------------

namespace taosim
{

struct InstructionLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<InstructionLogContext>;
    using PayloadType = std::variant<
        std::shared_ptr<PlaceOrderMarketPayload>,
        std::shared_ptr<PlaceOrderLimitPayload>>;

    AgentId agentId;
    OrderID orderId;
    PayloadType payload;

    InstructionLogContext(
        AgentId agentId,
        OrderID orderId,
        const PayloadType& payload) noexcept
        : agentId{agentId}, orderId{orderId}, payload{payload}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;
};

//-------------------------------------------------------------------------

}  // namespace taosim

//-------------------------------------------------------------------------