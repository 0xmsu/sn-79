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
struct convert<Eigen::MatrixXd>
{
    const msgpack::object& operator()(const msgpack::object& o, Eigen::MatrixXd& v) const
    {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto& arr = o.via.array;

        const auto rows = arr.ptr[0].as<int>();
        const auto cols = arr.ptr[1].as<int>();

        v.resize(rows, cols);

        for (size_t i = 0; i < rows * cols; ++i) {
            v.data()[i] = arr.ptr[2 + i].as<double>();
        }

        return o;
    }
};

template<>
struct pack<Eigen::MatrixXd>
{
    template<typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, const Eigen::MatrixXd& v) const
    {
        o.pack_array(2 + v.size());

        o.pack(v.rows());
        o.pack(v.cols());

        for (size_t i = 0; i < v.size(); ++i) {
            o.pack(v.data()[i]);
        }

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------