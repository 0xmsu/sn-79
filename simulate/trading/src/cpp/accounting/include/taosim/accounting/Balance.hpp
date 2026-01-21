/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "FreeInfo.hpp"
#include "JsonSerializable.hpp"
#include "Loan.hpp"
#include "Order.hpp"
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

class Balance : public JsonSerializable
{
public:
    using Reservations = std::map<OrderID, decimal_t>;

    explicit Balance(
        decimal_t total = {},
        const std::string& symbol = {},
        uint32_t roundingDecimals = 4);

    ~Balance() noexcept = default;
    Balance(const Balance&) noexcept = default;
    Balance& operator=(const Balance&) noexcept = default;
    Balance(Balance&& other) noexcept;
    Balance& operator=(Balance&& other) noexcept;

    [[nodiscard]] auto&& getInitial(this auto&& self) noexcept { return self.m_initial; }
    [[nodiscard]] auto&& getFree(this auto&& self) noexcept { return self.m_free; }
    [[nodiscard]] auto&& getReserved(this auto&& self) noexcept { return self.m_reserved; }
    [[nodiscard]] auto&& getTotal(this auto&& self) noexcept { return self.m_total; }
    [[nodiscard]] auto&& getReservations(this auto&& self) noexcept { return self.m_reservations; }
    [[nodiscard]] auto&& getSymbol(this auto&& self) noexcept { return self.m_symbol; }
    [[nodiscard]] auto&& getRoundingDecimals(this auto&& self) noexcept { return self.m_roundingDecimals; }

    [[nodiscard]] FreeInfo canFree(OrderID id, std::optional<decimal_t> amount = {}) const noexcept;
    [[nodiscard]] bool canReserve(decimal_t amount) const noexcept;
    [[nodiscard]] std::optional<decimal_t> getReservation(OrderID id) const noexcept;

    void deposit(decimal_t amount, BookId bookId);
    decimal_t makeReservation(OrderID id, decimal_t amount, BookId bookId);
    decimal_t freeReservation(OrderID id, BookId bookId, std::optional<decimal_t> amount = {});
    decimal_t tryFreeReservation(OrderID orderId, BookId bookId, std::optional<decimal_t> amount = {});
    void voidReservation(OrderID id, BookId bookId, std::optional<decimal_t> amount = {});

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    friend std::ostream& operator<<(std::ostream& os, const Balance& bal) noexcept;

    [[nodiscard]] static Balance fromXML(pugi::xml_node node, uint32_t roundingDecimals);
    [[nodiscard]] static Balance fromJson(const rapidjson::Value& json);

private:
    void checkConsistency(std::source_location sl, BookId bookId);
    void move(Balance&& other) noexcept;

    [[nodiscard]] decimal_t roundAmount(decimal_t amount) const;
    [[nodiscard]] std::optional<decimal_t> roundAmount(std::optional<decimal_t> amount) const;   

    decimal_t m_initial{};
    decimal_t m_free{};
    decimal_t m_reserved{};
    decimal_t m_total{};
    Reservations m_reservations;
    std::string m_symbol;
    uint32_t m_roundingDecimals;
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::accounting::Balance>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::accounting::Balance& bal, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "{} ({} | {})",
            bal.getTotal(),
            bal.getFree(),
            bal.getReserved());
    }
};

//-------------------------------------------------------------------------
