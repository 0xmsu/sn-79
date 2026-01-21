/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/FundamentalPrice.hpp>
#include <taosim/checkpoint/serialization/process/RNG.hpp>
#include <taosim/serialization/msgpack/Eigen/MatrixXd.hpp>
#include <taosim/serialization/msgpack/Eigen/VectorXd.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::process::FundamentalPrice>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::process::FundamentalPrice& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        auto& s = v.state();

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "value") {
                val.convert(s.value);
            }
            else if (key == "dJ") {
                val.convert(s.dJ);
            }
            else if (key == "t") {
                val.convert(s.t);
            }
            else if (key == "W") {
                val.convert(s.W);
            }
            else if (key == "L") {
                val.convert(s.L);
            }
            else if (key == "X") {
                val.convert(s.X);
            }
            else if (key == "V") {
                val.convert(s.V);
            }
            else if (key == "BH") {
                val.convert(s.BH);
            }
            else if (key == "lastCount") {
                val.convert(s.lastCount);
            }
            else if (key == "lastSeed") {
                val.convert(s.lastSeed);
                v.rng()->seed(s.lastSeed);
            }
            else if (key == "lastSeedTime") {
                val.convert(s.lastSeedTime);
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::process::FundamentalPrice>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::FundamentalPrice& v) const
    {
        const auto& s = v.state();

        o.pack_map(11);

        o.pack("value");
        o.pack(s.value);

        o.pack("dJ");
        o.pack(s.dJ);

        o.pack("t");
        o.pack(s.t);

        o.pack("W");
        o.pack(s.W);

        o.pack("L");
        o.pack(s.L);

        o.pack("X");
        o.pack(s.X);

        o.pack("V");
        o.pack(s.V);

        o.pack("BH");
        o.pack(s.BH);

        o.pack("lastCount");
        o.pack(s.lastCount);

        o.pack("lastSeed");
        o.pack(s.lastSeed);

        o.pack("lastSeedTime");
        o.pack(s.lastSeedTime);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------