/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "json_util.hpp"
#include <taosim/mp/mp.hpp>

//-------------------------------------------------------------------------

namespace taosim::json
{

namespace
{

template<typename T>
concept IsL3SerializableValue =
    requires (T t, rapidjson::Document& json, const std::string& key) {
        { t.L3Serialize(json, key) };
    };

template<typename T>
concept IsL3SerializablePointer =
    (mp::IsPointer<T> && requires (T t, rapidjson::Document& json, const std::string& key) {
        { t->L3Serialize(json, key) };
    });

}  // namespace

template<typename T>
concept IsL3Serializable = IsL3SerializableValue<T> || IsL3SerializablePointer<T>;

}  // namespace taosim::json

//-------------------------------------------------------------------------