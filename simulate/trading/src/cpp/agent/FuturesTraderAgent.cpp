/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/FuturesTraderAgent.hpp>

#include <taosim/process/FuturesSignal.hpp>
#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "Simulation.hpp"

#include <boost/algorithm/string/regex.hpp>

#include <algorithm>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

FuturesTraderAgent::FuturesTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_rng = &simulation()->rng();

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

    attr = node.attribute("sigmaN");
    m_sigmaN = ( attr.empty() || attr.as_double() < 0.0f) ? 0.7 : attr.as_double();

    if (attr = node.attribute("sigmaEps"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaEps' should have a value greater than 0.0f", ctx));
    }
    m_sigmaEps = attr.as_double();

    if (attr = node.attribute("minOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'minLatency' should have a value greater than 0", ctx));
    }
    m_opl.min = attr.as_ullong();
    if (attr = node.attribute("maxOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'maxLatency' should have a value greater than 0", ctx));
    }
    m_opl.max = attr.as_ullong();
    if (m_opl.min >= m_opl.max) {
        throw std::invalid_argument(fmt::format(
            "{}: 'minOPLatency' ({}) should be strictly less 'maxOPLatency' ({})",
            ctx, m_opl.min, m_opl.max));
    }

    attr = node.attribute("volume");
    m_volume = (attr.empty() || attr.as_double() <= 0.0) ? 1.0 : attr.as_double();

    attr = node.attribute("tau");
    m_tau = (attr.empty() || attr.as_ullong() == 0) ? 120'000'000'000 : attr.as_ullong();

    
    attr = node.attribute("orderTypeProb");
    m_orderTypeProb = (attr.empty() || attr.as_float() <= 0.0f) ? 0.5f : attr.as_float();

    m_lastUpdate = std::vector<Timestamp>(m_bookCount,0);
    
    m_orderFlag = std::vector<bool>(m_bookCount, false);

    m_priceIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);

    m_marketFeedLatencyDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "MFLmean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "MFLstd";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };
    m_decisionMakingDelayDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "delayMean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "delaySTD";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };

    
    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution = std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();

    size_t pos = name().find_last_not_of("0123456789");
    if (pos != std::string::npos && pos + 1 < name().size()) {
        std::string numStr = name().substr(pos + 1);
        m_catUId = static_cast<uint32_t>(std::stoul(numStr));
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::receiveMessage(Message::Ptr msg)
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
        handleRetrieveL1Response(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_MARKET") {
        handleMarketOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    }
    else if (msg->type == "WAKEUP") {
        handleWakeup(msg);
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleSimulationStart()
{
    if (m_catUId == 0) {
        for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {

            auto chosenAgent = selectTurn();

            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                decisionMakingDelay(),
                name(),
                fmt::format("{}_{}", m_baseName, chosenAgent),
                "WAKEUP",
                MessagePayload::create<RetrieveL1Payload>(bookId));
        }
    }
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleSimulationStop()
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleTradeSubscriptionResponse()
{

}

//-------------------------------------------------------------------------

uint64_t FuturesTraderAgent::selectTurn() {
    const auto& agentBaseNamesToCounts = simulation()->localAgentManager()->roster()->baseNamesToCounts();
    return  std::uniform_int_distribution<uint64_t>{0, agentBaseNamesToCounts.at(m_baseName) - 1}(*m_rng);
}

//------------------------------------------------------------------------

void FuturesTraderAgent::handleWakeup(Message::Ptr &msg)
{
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        marketFeedLatency(),
        name(),
        m_exchange,
        "RETRIEVE_L1",
        msg->payload);
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleRetrieveL1Response(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(msg->payload);

    const BookId bookId = payload->bookId;

    uint64_t chosenOne = selectTurn();
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        decisionMakingDelay(),
        name(),
        fmt::format("{}_{}", m_baseName, chosenOne),
        "WAKEUP",
        MessagePayload::create<RetrieveL1Payload>(bookId));


    if (m_orderFlag.at(bookId)) return;
    

    const double bestBid = taosim::util::decimal2double(payload->bestBidPrice);
    const double bestAsk = taosim::util::decimal2double(payload->bestAskPrice);
    placeOrder(bookId, bestAsk, bestBid);
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleMarketOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(msg->payload);
    m_orderFlag.at(payload->requestPayload->bookId) = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleMarketOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::dynamic_pointer_cast<PlaceOrderMarketErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag.at(bookId) = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
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

void FuturesTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag.at(bookId) = false;
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void FuturesTraderAgent::handleTrade(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
}


//-------------------------------------------------------------------------

void FuturesTraderAgent::placeOrder(BookId bookId, double bestAsk, double bestBid)
{
    const auto futuresDetails = getProcessDetails(bookId, "external");
    const double logReturn = futuresDetails.logReturn; 
    if (logReturn == 0.0 || futuresDetails.volumeFactor == 0.0) return;
    const double sign = logReturn < 0.0 ? -1.0 : 1.0;
    const double epsilon = std::normal_distribution{0.0, m_sigmaEps}(*m_rng);
    const double forecast =  sign + epsilon;
    
    const float newMean = std::log(m_volume) * futuresDetails.volumeFactor;
    std::lognormal_distribution<> lognormalDist(newMean, 1); 
    double volume = lognormalDist(*m_rng);
    const double priceShift =  volume > 1.0 ? std::floor(volume)*m_priceIncrement : -1*std::floor(volume/m_priceIncrement)*m_priceIncrement;
    volume =  std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume == 0) return;
    const bool draw = std::bernoulli_distribution{m_orderTypeProb}(*m_rng);
    if (forecast > 0) {
        if (draw) {
            placeBuy(bookId, volume);
        } else {
            placeBid(bookId, volume, bestBid - priceShift);
        }
    }
    else if (forecast < 0) {
        if (draw) {
            placeSell(bookId, volume);
        } else {
            placeAsk(bookId, volume, bestAsk + priceShift);
        }
    }
}
//-------------------------------------------------------------------------

void FuturesTraderAgent::placeBid(BookId bookId, double volume, double price)
{
    m_orderFlag.at(bookId) = true;

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::BUY,
            taosim::util::double2decimal(volume),
            taosim::util::double2decimal(price),
            bookId));
}
//-------------------------------------------------------------------------

void FuturesTraderAgent::placeBuy(BookId bookId, double volume)
{
    m_orderFlag.at(bookId) = true;

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::BUY,
            taosim::util::double2decimal(volume),
            bookId));
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::placeAsk(BookId bookId, double volume, double price)
{
    m_orderFlag.at(bookId) = true;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::SELL,
            taosim::util::double2decimal(volume),
            taosim::util::double2decimal(price),
            bookId));
}

//-------------------------------------------------------------------------

void FuturesTraderAgent::placeSell(BookId bookId, double volume)
{
    m_orderFlag.at(bookId) = true;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        orderPlacementLatency(),
        name(),
        m_exchange,
        "PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::SELL,
            taosim::util::double2decimal(volume),
            bookId));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::orderPlacementLatency()
{
    return static_cast<Timestamp>(
        std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::marketFeedLatency()
{
    return static_cast<Timestamp>(std::min(std::abs(m_marketFeedLatencyDistribution(*m_rng)),
            m_marketFeedLatencyDistribution.mean() + 3 * m_marketFeedLatencyDistribution.stddev()));
}

//-------------------------------------------------------------------------

Timestamp FuturesTraderAgent::decisionMakingDelay()
{
    return static_cast<Timestamp>(std::min(std::abs(m_decisionMakingDelayDistribution(*m_rng)),
            m_decisionMakingDelayDistribution.mean()
            + 3.0 * m_decisionMakingDelayDistribution.stddev()));
}

//-------------------------------------------------------------------------

double FuturesTraderAgent::getProcessValue(BookId bookId, const std::string& name)
{
    return simulation()->exchange()->process(name, bookId)->value();
}

//-------------------------------------------------------------------------

FuturesTraderAgent::FuturesDetails FuturesTraderAgent::getProcessDetails(
    BookId bookId, const std::string& name)
{
    const auto externalProcess =
        dynamic_cast<process::FuturesSignal*>(simulation()->exchange()->process(name, bookId));    
    return {
        .logReturn = externalProcess->state().logReturn,
        .volumeFactor = externalProcess->state().volumeFactor
    };
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------