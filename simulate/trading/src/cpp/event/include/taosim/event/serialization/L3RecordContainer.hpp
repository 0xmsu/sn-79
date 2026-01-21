/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/L3RecordContainer.hpp>
#include <taosim/event/serialization/common.hpp>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::event::L3Record>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::event::L3Record& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        using T = std::remove_cvref_t<decltype(v.entries())>;
        v.entries() = o.as<T>();

        return o;
    }
};

template<>
struct pack<taosim::event::L3Record>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::L3Record& v) const
    {
        o.pack_array(v.size());

        for (const auto& entry : v) {
            o.pack(entry);
        }

        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::event::L3RecordContainer>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::event::L3RecordContainer& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        using T = std::remove_cvref_t<decltype(v.underlying())>;
        v.underlying() = o.as<T>();

        return o;
    }
};

template<>
struct pack<taosim::event::L3RecordContainer>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::L3RecordContainer& v) const
    {
        if constexpr (std::same_as<Stream, taosim::serialization::HumanReadableStream>) {
            o.pack_map(v.underlying().size());

            for (const auto& [bookId, record] : views::enumerate(v.underlying())) {
                o.pack(std::to_string(bookId));
                o.pack(record);
            }
        }
        else if constexpr (std::same_as<Stream, taosim::serialization::BinaryStream>) {
            o.pack(v.underlying());
        }
        else {
            static_assert(false, "Unrecognized Stream type");
        }

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
