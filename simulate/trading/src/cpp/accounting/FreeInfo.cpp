/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/FreeInfo.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

std::string FreeInfo::toString() const noexcept
{
    switch (status) {
        case FreeStatus::FREEABLE:
            return fmt::format("Order #{} is freeable for {}", orderId, amount.value());
        case FreeStatus::NEGATIVE_AMOUNT:
            return fmt::format(
                "Attempt freeing negative amount of {} for order #{}", amount.value(), orderId);
        case FreeStatus::AMOUNT_EXCEEDS_RESERVATION:
            return fmt::format(
                "Attempt freeing amount of {} exceeding reservation of {} for order #{}",
                amount.value(),
                reservation.value(),
                orderId);
        case FreeStatus::NONEXISTENT_RESERVATION:
            return fmt::format(
                "Attempt freeing {} for nonexistent order #{}", amount.value(), orderId);
        case FreeStatus::NONEXISTENT_RESERVATION_AND_AMOUNT:
            return fmt::format("Nonexistent reservation for order #{} and empty amount", orderId);
        case FreeStatus::NONEXISTENT_RESERVATION_AND_NEGATIVE_AMOUNT:
            return fmt::format(
                "Attempt freeing negative amount of {} for nonexistent reservation #{}",
                amount.value(),
                orderId);
        default:
            return fmt::format("Unknown free status code {}", std::to_underlying(status));
    }
}

//-------------------------------------------------------------------------

FreeException::FreeException(std::string msg) noexcept
    : m_msg{std::move(msg)}
{}

//-------------------------------------------------------------------------

const char* FreeException::what() const noexcept
{
    return m_msg.c_str();
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
