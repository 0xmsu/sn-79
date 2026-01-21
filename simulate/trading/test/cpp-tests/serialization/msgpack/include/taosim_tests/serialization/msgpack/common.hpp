/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace taosim_tests::serialization::msgpack
{

template<typename T>
[[nodiscard]] T performRoundTrip(const T& value)
{
    taosim::serialization::BinaryStream stream;
    ::msgpack::packer packer{stream};
    packer.pack(value);
    ::msgpack::object_handle objHandle = ::msgpack::unpack(stream.data(), stream.size());
    ::msgpack::object obj = objHandle.get();
    return obj.as<T>();
}

}  // taosim_tests::serialization::msgpack

//-------------------------------------------------------------------------