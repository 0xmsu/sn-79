/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "JsonSerializable.hpp"
#include <taosim/message/MessagePayload.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

struct Message : public JsonSerializable
{
    using Ptr = std::shared_ptr<Message>;

    Timestamp occurrence;
    Timestamp arrival;
    std::string source;
    std::vector<std::string> targets;
    std::string type;
    MessagePayload::Ptr payload;

    static constexpr char s_targetDelim = '|';

    Message() noexcept = default;

    Message(
        Timestamp occurrence,
        Timestamp arrival,
        const std::string& source,
        const std::vector<std::string>& targets,
        const std::string& type,
        MessagePayload::Ptr payload) noexcept
        : occurrence{occurrence},
          arrival{arrival},
          source{source},
          targets{targets},
          type{type},
          payload{payload}
    {}

    Message(
        Timestamp,
        Timestamp,
        const std::string&,
        const std::string&,
        const std::string&,
        MessagePayload::Ptr) noexcept;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<Message, Args...> && requires { typename Message::Ptr; }
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return typename Message::Ptr{new Message(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJsonMessage(const rapidjson::Value& json) noexcept;
    [[nodiscard]] static Ptr fromJsonResponse(
        const rapidjson::Value& json, Timestamp timestamp, const std::string& source) noexcept;
};

//-------------------------------------------------------------------------

struct CompareArrival
{
    bool operator()(Message::Ptr a, Message::Ptr b) { return a->arrival > b->arrival; }
};

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<Message>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const Message& msg, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(), "{}", taosim::json::jsonSerializable2str(msg));
    }
};

//-------------------------------------------------------------------------