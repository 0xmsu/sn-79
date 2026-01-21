/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Balance.hpp>
#include <taosim/accounting/Balances.hpp>
#include <taosim/accounting/serialization/Balance.hpp>
#include <taosim/accounting/serialization/Loan.hpp>
#include <taosim/accounting/serialization/RoundParams.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::Balances>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::Balances& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        auto convertLoans = [&](const msgpack::object& o) {
            if (o.type != msgpack::type::ARRAY) {
                throw taosim::serialization::MsgPackError{};
            }
            for (const auto& val : o.via.array) {
                if (val.type != msgpack::type::MAP) {
                    throw taosim::serialization::MsgPackError{};
                }
                const auto id = taosim::serialization::msgpackFindMap<OrderID>(val, "id");
                if (!id) {
                    throw taosim::serialization::MsgPackError{};
                }
                v.m_loans[*id] = val.as<taosim::accounting::Loan>();
            }
        };

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "baseLoan") {
                v.m_baseLoan = val.as<taosim::decimal_t>();
            }
            else if (key == "quoteLoan") {
                v.m_quoteLoan = val.as<taosim::decimal_t>();
            }
            else if (key == "baseCollateral") {
                v.m_baseCollateral = val.as<taosim::decimal_t>();
            }
            else if (key == "quoteCollateral") {
                v.m_quoteCollateral = val.as<taosim::decimal_t>();
            }
            else if (key == "base") {
                v.base = val.as<taosim::accounting::Balance>();
            }
            else if (key == "quote") {
                v.quote = val.as<taosim::accounting::Balance>();
            }
            else if (key == "buyLeverages") {
                val.convert(v.m_buyLeverages);
            }
            else if (key == "sellLeverages") {
                val.convert(v.m_sellLeverages);
            }
            else if (key == "loans") {
                convertLoans(val);
            }
            else if (key == "roundParams") {
                val.convert(v.m_roundParams);
                v.m_baseDecimals = v.m_roundParams.baseDecimals;
                v.m_quoteDecimals = v.m_roundParams.quoteDecimals;
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::accounting::Balances>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::Balances& v) const
    {
        o.pack_map(10);

        o.pack("baseLoan");
        o.pack(v.m_baseLoan);

        o.pack("quoteLoan");
        o.pack(v.m_quoteLoan);

        o.pack("baseCollateral");
        o.pack(v.m_baseCollateral);

        o.pack("quoteCollateral");
        o.pack(v.m_quoteCollateral);

        o.pack("base");
        o.pack(v.base);

        o.pack("quote");
        o.pack(v.quote);

        o.pack("buyLeverages");
        o.pack(v.m_buyLeverages);

        o.pack("sellLeverages");
        o.pack(v.m_sellLeverages);

        o.pack("loans");
        o.pack_array(v.m_loans.size());
        for (const auto& [id, loan] : v.m_loans) {
            o.pack_map(6);

            o.pack("id");
            o.pack(id);

            o.pack("amount");
            o.pack(loan.amount());

            o.pack("direction");
            o.pack(loan.direction());

            o.pack("leverage");
            o.pack(loan.leverage());

            o.pack("collateral");
            o.pack(loan.collateral());

            o.pack("marginCallPrice");
            o.pack(loan.marginCallPrice());
        }

        o.pack("roundParams");
        o.pack(v.m_roundParams);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
