/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>
#include "JsonSerializable.hpp"
#include "common.hpp"
#include <taosim/mp/mp.hpp>
#include "util.hpp"

#include <msgpack.hpp>

//-------------------------------------------------------------------------

struct MessagePayload : public JsonSerializable
{
    using Ptr = std::shared_ptr<MessagePayload>;

    MessagePayload() noexcept = default;

    virtual ~MessagePayload() noexcept = default;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override {};

    template<typename T, typename... Args>
    requires(
        std::derived_from<T, MessagePayload> && std::constructible_from<T, Args...> &&
        taosim::mp::IsPointer<typename T::Ptr>)
    [[nodiscard]] static T::Ptr create(Args&&... args) noexcept
    {
        return typename T::Ptr{new T(std::forward<Args>(args)...)};
    }
};

//-------------------------------------------------------------------------

struct ErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ErrorResponsePayload>;

    std::string message;

    ErrorResponsePayload() noexcept = default;

    ErrorResponsePayload(std::string message) : message{std::move(message)} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(message);
};

//-------------------------------------------------------------------------

struct SuccessResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<SuccessResponsePayload>;

    std::string message;

    SuccessResponsePayload(std::string message) : message{std::move(message)} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(message);
};

//-------------------------------------------------------------------------

struct EmptyPayload : public MessagePayload
{
    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;
};

//-------------------------------------------------------------------------

struct GenericPayload : public MessagePayload, public std::map<std::string, std::string>
{
    GenericPayload(std::map<std::string, std::string> initMap)
        : MessagePayload{}, map{std::move(initMap)}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);
};

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<MessagePayload>
{
    const msgpack::object& operator()(const msgpack::object& o, [[maybe_unused]] MessagePayload& v)
    {
        if (!o.is_nil()) {
            throw taosim::serialization::MsgPackError{};
        }
        return o;
    }
};

template<>
struct pack<MessagePayload>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const MessagePayload& v) const
    {
        o.pack_nil();
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
