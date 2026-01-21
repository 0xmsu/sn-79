/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>

#include <fmt/ranges.h>

#include <functional>
#include <optional>

//-------------------------------------------------------------------------

namespace taosim::serialization
{

//-------------------------------------------------------------------------

template<typename T>
concept MsgPackObjectType = std::same_as<std::remove_cvref_t<T>, msgpack::object>;

template<typename T>
using ReferenceWrapperTypeFromUniversalReferenceType =
    std::reference_wrapper<
        std::conditional_t<
            std::is_const_v<std::remove_reference_t<T>>
            || std::is_rvalue_reference_v<T>,
            const std::remove_reference_t<T>,
            std::remove_reference_t<T>
        >
    >;

//-------------------------------------------------------------------------

template<MsgPackObjectType T>
[[nodiscard]] auto msgpackFindMapObj(T&& haystack, std::string_view needle)
    -> std::optional<ReferenceWrapperTypeFromUniversalReferenceType<decltype(haystack)>>
{
    if (haystack.type != msgpack::type::MAP) {
        return {};
    }
    for (auto&& [k, val] : haystack.via.map) {
        auto key = k.template as<std::string_view>();
        if (key == needle) {
            using H = decltype(haystack);
            if constexpr (std::is_const_v<H> || std::is_rvalue_reference_v<H>) {
                return std::make_optional(std::cref(val));
            } else {
                return std::make_optional(std::ref(val));
            }
        }
    }
    return {};
}

//-------------------------------------------------------------------------

template<typename T>
[[nodiscard]] std::optional<T> msgpackFindMap(
    const msgpack::object& haystack, std::string_view needle)
{
    if (haystack.type != msgpack::type::MAP) {
        return {};
    }
    for (const auto& [k, val] : haystack.via.map) {
        auto key = k.as<std::string_view>();
        if (key == needle) {
            return std::make_optional(val.as<T>());
        }
    }
    return {};
}

//-------------------------------------------------------------------------

[[nodiscard]] inline std::string msgpackMapKeysToString(
    const msgpack::object& o, std::source_location sl = std::source_location::current())
{
    if (o.type != msgpack::type::MAP) {
        throw MsgPackError{sl};
    }
    std::vector<std::string> keys;
    for ([[maybe_unused]] const auto& [key, _] : o.via.map) {
        keys.push_back(key.as<std::string>());
    }
    return fmt::format("{}", fmt::join(keys, ", "));
}

//-------------------------------------------------------------------------

}  // namespace taosim::serialization

//-------------------------------------------------------------------------