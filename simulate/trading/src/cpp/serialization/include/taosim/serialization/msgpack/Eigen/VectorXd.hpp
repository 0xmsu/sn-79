/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/serialization/msgpack/common.hpp>

#include <Eigen/Core>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<Eigen::VectorXd>
{
    const msgpack::object& operator()(const msgpack::object& o, Eigen::VectorXd& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        v.resize(arr.size);

        for (size_t i = 0; i < arr.size; ++i) {
            v[i] = arr.ptr[i].as<double>();
        }

        return o;
    }
};

template<>
struct pack<Eigen::VectorXd>
{
    template<typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, const Eigen::VectorXd& v) const
    {
        o.pack_array(v.size());

        for (size_t i = 0; i < v.size(); ++i) {
            o.pack(v[i]);
        }

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------