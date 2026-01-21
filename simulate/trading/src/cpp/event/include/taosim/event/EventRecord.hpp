/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "JsonSerializable.hpp"
#include <taosim/mp/mp.hpp>
#include "util.hpp"

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

template<typename... Ts>
class EventRecord : public JsonSerializable
{
public:
    using Entry = std::variant<Ts...>;

    [[nodiscard]] size_t size() const noexcept { return m_entries.size(); }

    void push(Entry entry) noexcept { m_entries.push_back(std::move(entry)); }
    void clear() noexcept { m_entries.clear(); }

    [[nodiscard]] decltype(auto) begin(this auto&& self) { return self.m_entries.begin(); }
    [[nodiscard]] decltype(auto) end(this auto&& self) { return self.m_entries.end(); }

    [[nodiscard]] auto&& entries(this auto&& self) noexcept { return self.m_entries; }

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const
    {
        auto serialize = [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto& entry : m_entries) {
                std::visit(
                    [&](auto&& entry) {
                        using T = std::remove_cvref_t<decltype(entry)>;
                        rapidjson::Document entryJson{&allocator};
                        if constexpr (taosim::mp::IsPointer<T>) {
                            entry->jsonSerialize(entryJson);
                        } else {
                            entry.jsonSerialize(entryJson);
                        }
                        json.PushBack(entryJson, allocator);
                    },
                    entry);
            }
            if (json.Size() == 0) {
                json.SetNull();
            }
        };
        taosim::json::serializeHelper(json, key, serialize);
    }

private:
    std::vector<Entry> m_entries;
};

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------
