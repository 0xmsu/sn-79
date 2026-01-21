/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim
{

//-------------------------------------------------------------------------

enum class STPFlag : uint32_t
{
    NONE,
    CO, // Cancel the resting
    CN, // Cancel the aggressing
    CB, // Cancel both
    DC  // Decrement and Cancel
};

//-------------------------------------------------------------------------

enum class TimeInForce : uint32_t
{
    GTC,
    GTT,
    IOC,
    FOK
};

//-------------------------------------------------------------------------

enum class SettleType: int32_t 
{
    NONE = -2,
    FIFO = -1
};

using SettleFlag = std::variant<OrderID, SettleType>;

//-------------------------------------------------------------------------

}  // namespace taosim

//-------------------------------------------------------------------------

MSGPACK_ADD_ENUM(taosim::STPFlag);
MSGPACK_ADD_ENUM(taosim::TimeInForce);

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct pack<taosim::SettleType>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, taosim::SettleType v) const
    {
        return o.pack(std::to_underlying(v));
    }
};

template<>
struct convert<taosim::SettleType>
{
    msgpack::object const& operator()(msgpack::object const& o, taosim::SettleType& v) const
    {
        std::underlying_type_t<taosim::SettleType> tmp;
        o.convert(tmp);
        v = static_cast<taosim::SettleType>(tmp);
        return o;
    }
};

template<>
struct convert<taosim::SettleFlag>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::SettleFlag& v)
    {
        if (o.type == msgpack::type::NEGATIVE_INTEGER) {
            v = magic_enum::enum_cast<taosim::SettleType>(o.as<int32_t>()).value();
        }
        else if (o.type == msgpack::type::POSITIVE_INTEGER) {
            v = o.as<OrderID>();
        }
        else {
            throw taosim::serialization::MsgPackError{};
        }
        return o;
    }
};

template<>
struct pack<taosim::SettleFlag>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::SettleFlag& v) const
    {
        std::visit([&](auto&& settleFlag) { o.pack(settleFlag); }, v);
        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------