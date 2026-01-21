/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "json_util.hpp"
#include <taosim/mp/mp.hpp>

#include <optional>

//-------------------------------------------------------------------------
// TODO: Make this a C++20 concept instead.

class JsonSerializable
{
public:
    virtual ~JsonSerializable() noexcept = default;

    virtual void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const = 0;

protected:
    JsonSerializable() noexcept = default;
};

//-------------------------------------------------------------------------

namespace taosim::json
{

namespace
{

template<typename T>
concept IsJsonSerializableValue =
    requires (T t, rapidjson::Document& json, const std::string& key) {
        { t.jsonSerialize(json, key) };
    };

template<typename T>
concept IsJsonSerializablePointer =
    (mp::IsPointer<T> && requires (T t, rapidjson::Document& json, const std::string& key) {
        { t->jsonSerialize(json, key) };
    });

}  // namespace

template<typename T>
concept IsJsonSerializable = IsJsonSerializableValue<T> || IsJsonSerializablePointer<T>;

[[nodiscard]] std::string jsonSerializable2str(
    const IsJsonSerializable auto& serializable, const FormatOptions& formatOptions = {})
{
    rapidjson::Document json;
    if constexpr (mp::IsPointer<decltype(serializable)>) {
        serializable->jsonSerialize(json);
    } else {
        serializable.jsonSerialize(json);
    }
    return json2str(json, formatOptions);
}

}  // namespace taosim::json

//-------------------------------------------------------------------------
