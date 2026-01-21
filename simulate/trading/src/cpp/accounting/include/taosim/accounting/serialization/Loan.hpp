/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Loan.hpp>
#include <taosim/accounting/serialization/Collateral.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::Loan>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::Loan& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        taosim::accounting::LoanDesc loanDesc;

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "amount") {
                loanDesc.amount = val.as<taosim::decimal_t>();
            }
            else if (key == "direction") {
                loanDesc.direction = val.as<OrderDirection>();
            }
            else if (key == "leverage") {
                loanDesc.leverage = val.as<taosim::decimal_t>();
            }
            else if (key == "collateral") {
                loanDesc.collateral = val.as<taosim::accounting::Collateral>();
            }
            else if (key == "marginCallPrice") {
                loanDesc.marginCallPrice = val.as<taosim::decimal_t>();
            }
        }

        v = taosim::accounting::Loan(std::move(loanDesc));

        return o;
    }
};

template<>
struct pack<taosim::accounting::Loan>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::Loan& v) const
    {
        o.pack_map(5);

        o.pack("amount");
        o.pack(v.amount());

        o.pack("direction");
        o.pack(v.direction());

        o.pack("leverage");
        o.pack(v.leverage());

        o.pack("collateral");
        o.pack(v.collateral());

        o.pack("marginCallPrice");
        o.pack(v.marginCallPrice());
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------