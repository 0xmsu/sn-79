/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/event/L3RecordContainer.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

void L3RecordContainer::clear() noexcept
{
    for (auto& individualRecord : m_underlying) {
        individualRecord.clear();
    }
}

//-------------------------------------------------------------------------

void L3RecordContainer::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        for (const auto& [bookId, record] : views::enumerate(m_underlying)) {
            record.jsonSerialize(json, std::to_string(bookId));
        }
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------
