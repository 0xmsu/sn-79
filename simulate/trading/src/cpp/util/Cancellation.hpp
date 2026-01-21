/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/Cancellation.hpp>
#include <taosim/event/serialization/Cancellation.hpp>
#include "CheckpointSerializable.hpp"
#include "JsonSerializable.hpp"
#include "Order.hpp"
#include "common.hpp"
#include "json_util.hpp"

#include <memory>
#include <optional>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

struct CancellationLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<CancellationLogContext>;

    AgentId agentId;
    BookId bookId;
    Timestamp timestamp;

    CancellationLogContext() noexcept = default;

    CancellationLogContext(AgentId agentId, BookId bookId, Timestamp timestamp)
        : agentId{agentId}, bookId{bookId}, timestamp{timestamp}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(agentId, bookId, timestamp);
};

//-------------------------------------------------------------------------

struct CancellationWithLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<CancellationWithLogContext>;

    taosim::event::Cancellation cancellation;
    CancellationLogContext::Ptr logContext;

    CancellationWithLogContext() noexcept = default;

    CancellationWithLogContext(
        taosim::event::Cancellation cancellation,
        CancellationLogContext::Ptr logContext) noexcept
        : cancellation{cancellation}, logContext{logContext}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(cancellation, logContext);
};

//-------------------------------------------------------------------------
