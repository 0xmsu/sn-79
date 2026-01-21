/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/RandomTraderAgent.hpp>

#include <taosim/event/Cancellation.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <Simulation.hpp>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

RandomTraderAgent::RandomTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void RandomTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    pugi::xml_attribute attr;
    static constexpr auto ctx = std::source_location::current().function_name();

    if (attr = node.attribute("exchange"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'exchange'", ctx));
    }
    m_exchange = attr.as_string();

    if (simulation()->exchange() == nullptr) {
        throw std::runtime_error(fmt::format(
            "{}: exchange must be configured a priori", ctx));
    }
    m_bookCount = simulation()->exchange()->books().size();
    m_topLevel = std::vector<TopLevel>(m_bookCount, TopLevel{});
    m_orderFlag = std::vector<bool>(m_bookCount, false);

    if (attr = node.attribute("tau"); attr.empty() || attr.as_ullong(120'000'000'000) == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tau' should have a value greater than 0", ctx));
    }
    m_tau = attr.as_ullong(120'000'000'000);
    m_quantityMin = node.attribute("minQuantity").as_double(0.01); 
    m_quantityMax = node.attribute("maxQuantity").as_double(2.0); 
}

//-------------------------------------------------------------------------

void RandomTraderAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (msg->type == "EVENT_SIMULATION_END") {
        handleSimulationStop();
    }
    else if (msg->type == "RESPONSE_SUBSCRIBE_EVENT_TRADE") {
        handleTradeSubscriptionResponse();
    }
    else if (msg->type == "RESPONSE_RETRIEVE_L1") {
        handleRetrieveResponse(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersErrorResponse(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    }
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleSimulationStart()
{
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        1,
        name(),
        m_exchange,
        "SUBSCRIBE_EVENT_TRADE");
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleSimulationStop()
{}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleTradeSubscriptionResponse()
{
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            //Should take from gracePeriod
            600'000'000'000,
            name(),
            m_exchange,
            "RETRIEVE_L1",
            MessagePayload::create<RetrieveL1Payload>(bookId));
    }
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleRetrieveResponse(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(msg->payload);
    BookId bookId = payload->bookId;

    std::random_device rd;
    std::mt19937 gen(rd());
    double quantityBid = std::uniform_real_distribution<double>{m_quantityMin, m_quantityMax}(gen);
    double quantityAsk = std::uniform_real_distribution<double>{m_quantityMin, m_quantityMax}(gen);

    OrderDirection direction; 
    double bestAsk = util::decimal2double(payload->bestAskPrice);
    double bestBid = util::decimal2double(payload->bestBidPrice);
    double limitBidPrice = std::uniform_real_distribution<double>{bestBid,bestAsk}(gen);
    double limitAskPrice = std::uniform_real_distribution<double>{limitBidPrice, bestAsk}(gen);

    // add later for testing
    double leverage = 0.0;

    sendOrder(bookId, OrderDirection::BUY, quantityBid, limitBidPrice, leverage);    
    sendOrder(bookId, OrderDirection::SELL, quantityAsk, limitAskPrice, leverage);    
    
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        // Should take from step
        1'000'000'000,
        name(),
        m_exchange,
        "RETRIEVE_L1",
        MessagePayload::create<RetrieveL1Payload>(bookId));
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
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

    m_orderFlag.at(payload->requestPayload->bookId) = false;
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag.at(bookId) = false;
}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void RandomTraderAgent::handleTrade(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void RandomTraderAgent::sendOrder(BookId bookId, OrderDirection direction,
    double volume, double price, double leverage) {

    m_orderFlag.at(bookId) = true;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> delayDist{1'500.0f,500.0f};

    float min_delay = 10'000'000.0f; 
    float max_delay = 1'000'000'000.0f; 
    float t = std::clamp(std::abs(delayDist(gen))/3'000.0f,0.0f,1.0f);
    float exp_scale = 5.0f;
    float delay_frac = (std::exp(exp_scale * t) - 1) / (std::exp(exp_scale)-1);
    Timestamp delay = (min_delay + delay_frac * (max_delay - min_delay));
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        delay,
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(
            direction,
            taosim::util::double2decimal(volume),
            taosim::util::double2decimal(price),
            taosim::util::double2decimal(leverage),
            bookId));
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------