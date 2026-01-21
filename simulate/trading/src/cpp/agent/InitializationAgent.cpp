/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "InitializationAgent.hpp"

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/accounting/Account.hpp>
#include "Simulation.hpp"

#include <cmath>

//-------------------------------------------------------------------------

InitializationAgent::InitializationAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void InitializationAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    static constexpr auto ctx = std::source_location::current().function_name();

    m_rng = &simulation()->rng();

    pugi::xml_attribute attr;

    if (attr = node.attribute("exchange"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: Missing required attribute 'exchange'", ctx));
    }
    m_exchange = attr.as_string();

    if (simulation()->exchange() == nullptr) {
        throw std::runtime_error(fmt::format(
            "{}: Exchange must be configured a priori", ctx));
    }
    m_bookCount = simulation()->exchange()->books().size();

    m_price = taosim::util::decimal2double(simulation()->exchange()->config2().initialPrice);

    if (attr = node.attribute("tau"); attr.empty() || attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: Missing required attribute 'tau'", ctx));
    }
    m_tau = attr.as_ullong();    

    m_priceIncrement =
        1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement =
        1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);
}

//-------------------------------------------------------------------------

void InitializationAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "EVENT_SIMULATION_START") {
        placeBuyOrders();
        placeSellOrders();
    } else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    } 
}

//-------------------------------------------------------------------------

void InitializationAgent::placeBuyOrders()
{
    const auto& account = simulation()->account(name());

    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const double freeQuote = taosim::util::decimal2double(account.at(bookId).quote.getFree());
        const double maxQuantity = freeQuote / m_price / 2;
        double usedQuote = 0.0;
        while (usedQuote < freeQuote) {
            const double price = [&] {
                double price = std::uniform_real_distribution{0.0, m_price - m_priceIncrement}(*m_rng);
                return std::floor(price / m_priceIncrement) * m_priceIncrement;
            }();
            const double quantity = [&] {
                double quantity = std::min(
                    std::uniform_real_distribution{0.0, maxQuantity}(*m_rng),
                    (freeQuote - usedQuote) / price);
                return std::floor(quantity / m_volumeIncrement) * m_volumeIncrement;
            }();
            if (quantity <= 0.0) break;
            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                0,
                name(),
                m_exchange,
                "PLACE_ORDER_LIMIT",
                MessagePayload::create<PlaceOrderLimitPayload>(
                    OrderDirection::BUY,
                    taosim::util::double2decimal(quantity),
                    taosim::util::double2decimal(price),
                    bookId));
            usedQuote += price * quantity;
        }
    }
}

//-------------------------------------------------------------------------

void InitializationAgent::placeSellOrders()
{
    const auto& account = simulation()->account(name());

    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const double freeBase = taosim::util::decimal2double(account.at(bookId).base.getFree());
        const double maxQuantity = freeBase / 2;
        double usedBase = 0.0;
        while (usedBase < freeBase) {
            const double price = [&] {
                double price = std::uniform_real_distribution{
                    m_price + m_priceIncrement, m_price * 2}(*m_rng);
                return std::floor(price / m_priceIncrement) * m_priceIncrement;
            }();
            const double quantity = [&] {
                double quantity = std::min(
                    std::uniform_real_distribution{0.0, maxQuantity}(*m_rng),
                    freeBase - usedBase);
                return std::floor(quantity / m_volumeIncrement) * m_volumeIncrement;
            }();
            if (quantity <= 0.0) break;
            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                0,
                name(),
                m_exchange,
                "PLACE_ORDER_LIMIT",
                MessagePayload::create<PlaceOrderLimitPayload>(
                    OrderDirection::SELL,
                    taosim::util::double2decimal(quantity),
                    taosim::util::double2decimal(price),
                    bookId));
            usedBase += quantity;
        }
    }
}

//-------------------------------------------------------------------------

void InitializationAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        m_tau,
        name(),
        m_exchange,
        "CANCEL_ORDERS",
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation(payload->id)}, payload->requestPayload->bookId));
}

//-------------------------------------------------------------------------