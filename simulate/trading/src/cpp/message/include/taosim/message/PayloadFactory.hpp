/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

class PayloadFactory
{
public:
    [[nodiscard]] static MessagePayload::Ptr createFromJsonMessage(const rapidjson::Value& json);
    [[nodiscard]] static MessagePayload::Ptr createFromMessagePack(const msgpack::object& o, std::string_view type);

private:
    PayloadFactory() = default;
};

//-------------------------------------------------------------------------
