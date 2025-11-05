/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "StylizedTraderAgent.hpp"

#include "taosim/message/ExchangeAgentMessagePayloads.hpp"
#include "taosim/message/MessagePayload.hpp"
#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "Simulation.hpp"

#include <boost/algorithm/string/regex.hpp>
#include <boost/bimap.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/random.hpp>
#include <boost/random/laplace_distribution.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

#include <algorithm>

//-------------------------------------------------------------------------


inline auto investmentPosition = [](double price, double forecast, double variance, double base, double quote, double risk, double constant) {
    return std::log(forecast/price)/(variance*price) * (1/risk * (base*price + quote) + constant);
};    

namespace br = boost::random;
//-------------------------------------------------------------------------

StylizedTraderAgent::StylizedTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::configure(const pugi::xml_node& node)
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

    if (attr = node.attribute("sigmaF"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaF' should have a value of at least 0.0f", ctx));
    }
    const double sigmaF = attr.as_double();
    if (attr = node.attribute("sigmaC"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaC' should have a value of at least 0.0f", ctx));
    }
    const double sigmaC = attr.as_double();
    if (attr = node.attribute("sigmaN"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaN' should have a value of at least 0.0f", ctx));
    }
    const double sigmaN = attr.as_double();
    m_weight = {
        .F = std::abs(br::laplace_distribution{sigmaF, sigmaF}(*m_rng)),
        .C = std::abs(br::laplace_distribution{sigmaC, sigmaC}(*m_rng)),
        .N = std::abs(br::laplace_distribution{sigmaN, sigmaN}(*m_rng))
    };
    m_weightNormalizer = 1.0f / (m_weight.F + m_weight.C + m_weight.N);

    m_priceF0 = simulation()->exchange()->process("fundamental", BookId{})->value();

    m_price0 = taosim::util::decimal2double(simulation()->exchange()->config2().initialPrice);

    m_tauF = std::vector<double>(m_bookCount, attr.as_double()); 
    m_tauFOrig = attr.as_double();

    if (attr = node.attribute("sigmaEps"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaEps' should have a value greater than 0.0f", ctx));
    }
    m_sigmaEps = attr.as_double();
    m_hara = node.attribute("HARA").as_double();
    
    if (attr = node.attribute("r_aversion"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'r_aversion' should have a value greater than 0.0f", ctx));
    }
    // NOTE Modified inserted parameter is the one we want as the average!
    m_riskAversion0 = attr.as_double();
    const double riskAversionCoef = m_riskAversion0* (1.0f + sigmaC)/(1.0f + sigmaF);
    m_riskAversion = riskAversionCoef * (1.0f + m_weight.F) / (1.0f + m_weight.C);

    attr = node.attribute("volGuard");
    if (attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'volatility Guard (volGuard)' should have a value greater than 0.0f", ctx));
    }
    m_volatilityGuard =  attr.as_double();
    const float p_low  = m_volatilityGuard;
    const float p_high = 1- m_volatilityGuard;    
    const float L1 = std::log((1 - m_volatilityGuard) / m_volatilityGuard);
    const float L2 = std::log(m_volatilityGuard / (1- m_volatilityGuard));
    m_slopeVolGuard =  (L1 - L2) / (10*m_volatilityGuard - m_volatilityGuard/100);
    m_volGuardX0 = m_volatilityGuard/10 + L1/m_slopeVolGuard;

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

    m_price = m_priceF0;

    m_orderFlag = std::vector<bool>(m_bookCount, false);

    m_priceIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);

    m_debug = node.attribute("debug").as_bool();

    m_regimeSwitchKickback.resize(m_bookCount);
    m_regimeChangeFlag = node.attribute("regimeChangeFlag").as_bool();
    if (m_regimeChangeFlag) {
        if (attr = node.attribute("sigmaFRegime"); attr.empty() || attr.as_double() < 0.0f) {
            throw std::invalid_argument(fmt::format(
                "{}: attribute 'sigmaFRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaFRegime = attr.as_double();
        if (attr = node.attribute("sigmaCRegime"); attr.empty() || attr.as_double() < 0.0f) {
            throw std::invalid_argument(fmt::format(
                "{}: attribute 'sigmaCRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaCRegime = attr.as_double();
        if (attr = node.attribute("sigmaNRegime"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaNRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaNRegime = attr.as_double();
        m_weightRegime = {
            .F = std::abs(br::laplace_distribution{sigmaFRegime, sigmaFRegime}(*m_rng)),
            .C = std::abs(br::laplace_distribution{sigmaCRegime, sigmaCRegime}(*m_rng)),
            .N = std::abs(br::laplace_distribution{sigmaNRegime, sigmaNRegime}(*m_rng))
            };
        m_regimeChangeProb = std::vector<float>(m_bookCount, std::clamp(node.attribute("regimeProb").as_float(), 0.0f, 1.0f));
        if (attr = node.attribute("tauFRegime"); attr.empty() || attr.as_double() <= 0.0) {
            throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauFRegime' should have a value greater than 0.0", ctx));
        }
        m_tauFRegime = attr.as_double();
    } else {
        m_weightRegime = m_weight;
        m_regimeChangeProb = std::vector<float>(m_bookCount, 0.0f);
        m_tauFRegime = 1.0;
    }
    
    m_regimeState = std::vector<RegimeState>(m_bookCount,RegimeState::NORMAL);

    m_weightOrig = m_weight;


    if (attr = node.attribute("pO_alpha"); attr.empty() || attr.as_double() < 0.0f || attr.as_double() >= 1.0f){
        m_alpha = 0.0;
    } else {
        m_alpha = attr.as_double();
    }

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

    if (attr = node.attribute("tau"); attr.empty() || attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tau' should have a value greater than 0", ctx));
    }
    m_tau0 = attr.as_ullong();
    const double tauCoef = (m_tau0 *(m_decisionMakingDelayDistribution.mean() + m_marketFeedLatencyDistribution.mean())) * (1.0f + sigmaC)/(1.0f + sigmaF);
    m_tau = std::min(
        static_cast<Timestamp>(std::ceil(
            tauCoef * (1.0f + m_weight.F) / (1.0f + m_weight.C))),
        simulation()->duration() - 1);
    if (attr = node.attribute("tauF"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauF' should have a value greater than 0.0", ctx));
    }
    

    if (attr = node.attribute("tauHist"); attr.empty() || attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauHist' should have a value greater than 0", ctx));
    }

    m_tauHist = attr.as_ullong();
    const Timestamp averageStepsCoef = static_cast<Timestamp>( m_tau0 * (1.0f + sigmaC)/(1.0f + sigmaF)); 
    m_historySize = std::max(
        static_cast<Timestamp>(std::ceil(averageStepsCoef* (1.0 + m_weight.F) / (1.0 + m_weight.C))), m_tauHist);
    attr = node.attribute("GBM_X0");
    const double gbmX0 = (attr.empty() || attr.as_double() <= 0.0f) ?  0.001 : attr.as_double();
    attr = node.attribute("GBM_mu");
    const double gbmMu = (attr.empty() || attr.as_double() < 0.0f) ? 0 : attr.as_double();
    attr = node.attribute("GBM_sigma");
    const double gbmSigma = (attr.empty() || attr.as_double() < 0.0f) ? 0.1 : attr.as_double();
    attr = node.attribute("GBM_seed");
    const uint64_t gbmSeed = attr.as_ullong(10000); 

    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        m_topLevel.push_back(TopLevel{});
        GBMValuationModel gbmPrice{gbmX0, gbmMu, gbmSigma, gbmSeed + bookId + 1};
        const auto Xt = gbmPrice.generatePriceSeries(1, m_historySize);
        m_priceHist.push_back([&] {
            decltype(m_priceHist)::value_type hist{m_historySize};
            for (uint32_t i = 0; i < m_historySize; ++i) {
                hist.push_back(m_price0 * (1.0 + Xt[i]));
            }
            return hist;
        }());
        m_logReturns.push_back([&] {
            decltype(m_logReturns)::value_type logReturns{m_historySize};
            const auto& priceHist = m_priceHist.at(bookId);
            logReturns.push_back(Xt[0]);
            for (uint32_t i = 1; i < priceHist.capacity(); ++i) {
                logReturns.push_back(std::log(priceHist[i] / priceHist[i - 1]));
            }
            return logReturns;
        }());
    }

    m_tradePrice.resize(m_bookCount);
    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution =  std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 

    if (attr = node.attribute("scaleR"); attr.empty() || attr.as_double() <= 0.0) {
            throw std::invalid_argument{fmt::format(
                    "{}: Attribute 'scaleR' should be >= 0, was {}", ctx, attr.as_double())};
    } 
    const double scaleR = attr.as_double();
    m_rayleigh = std::make_unique<taosim::stats::RayleighDistribution>(scaleR);

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();

}

//-------------------------------------------------------------------------

void StylizedTraderAgent::receiveMessage(Message::Ptr msg)
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

void StylizedTraderAgent::handleSimulationStart()
{
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        1,
        name(),
        m_exchange,
        "SUBSCRIBE_EVENT_TRADE");
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleSimulationStop()
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleTradeSubscriptionResponse()
{
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            1,
            name(),
            m_exchange,
            "RETRIEVE_L1",
            MessagePayload::create<RetrieveL1Payload>(bookId));
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleRetrieveL1Response(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(msg->payload);

    const BookId bookId = payload->bookId;

    auto rng = std::mt19937{simulation()->currentTimestamp() + static_cast<Timestamp>(simulation()->bookIdCanon(bookId))};

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        static_cast<Timestamp>(std::min(
            std::abs(m_marketFeedLatencyDistribution(rng))
            + std::abs(m_decisionMakingDelayDistribution(rng)),
            m_marketFeedLatencyDistribution.mean()
            + m_decisionMakingDelayDistribution.mean()
            + 3.0 * (m_marketFeedLatencyDistribution.stddev()
                + m_decisionMakingDelayDistribution.stddev()))),
        name(),
        m_exchange,
        "RETRIEVE_L1",
        MessagePayload::create<RetrieveL1Payload>(bookId));

    auto& topLevel = m_topLevel.at(bookId);
    topLevel.bid = taosim::util::decimal2double(payload->bestBidPrice);
    topLevel.ask = taosim::util::decimal2double(payload->bestAskPrice);
    

    if  (topLevel.bid == 0.0) topLevel.bid = m_tradePrice.at(bookId).price;
    if  (topLevel.ask == 0.0) topLevel.ask = m_tradePrice.at(bookId).price;
    const double midQuote = 0.5 * (topLevel.bid + topLevel.ask);
    const double spotPrice = midQuote;
    const double lastPrice = midQuote;

    m_logReturns.at(bookId).push_back(
                    std::log(lastPrice / m_priceHist.at(bookId).back()));
    m_priceHist.at(bookId).push_back(lastPrice);

    const double askVol = taosim::util::decimal2double(payload->askTotalVolume);
    const double bidVol = taosim::util::decimal2double(payload->bidTotalVolume);
    const float volumeImbalance = (bidVol- askVol)/(bidVol + askVol);

    if (m_regimeState.at(bookId) == RegimeState::NORMAL && std::bernoulli_distribution{std::abs(volumeImbalance)}(rng)) {
        m_regimeChangeProb.at(bookId) = std::abs(volumeImbalance);
        updateRegime(bookId);
    } else if (m_regimeState.at(bookId) == RegimeState::REGIME_A) {
        updateRegime(bookId);
    }

    if (m_orderFlag.at(bookId)) return;

    
    auto linspace = [](double start, double stop, int num) -> std::vector<double> {
        if (!(num > 1)) {
            throw std::invalid_argument{fmt::format(
                "{}: parameter 'num' should be > 1, was {}",
                std::source_location::current().function_name(), num)};
        }
        const double step = (stop - start) / (num - 1);
        return views::iota(0, num)
            | views::transform([=](int k) { return start + k * step; })
            | ranges::to<std::vector>;
    };

    const uint32_t numActingAgents = [&] {
        const double rayleighDraw = m_rayleigh->sample(rng); 
        const auto bins = linspace(0.0, 5.0, 10);
        return std::upper_bound(bins.begin(), bins.end(), rayleighDraw) - bins.begin() - 1;
    }();

    const auto& agentBaseNamesToCounts =
        simulation()->localAgentManager()->roster()->baseNamesToCounts();

    const auto categoryIdToAgentType = [&] {
        boost::bimap<uint32_t, std::string> res;
        auto filteredBaseNames = agentBaseNamesToCounts
            | views::keys
            | views::filter([](const auto& baseName) { return baseName.contains("STYLIZED_TRADER_AGENT"); });
        for (const auto& [id, baseName] : views::enumerate(filteredBaseNames)) {
            res.insert({static_cast<uint32_t>(id), baseName});
        }
        return res;
    }();

    auto multinomial = [&] {
        const auto weights = agentBaseNamesToCounts
            | views::filter([](const auto& kv) { return kv.first.contains("STYLIZED_TRADER_AGENT"); })
            | views::values
            | ranges::to<std::vector>;
        return std::discrete_distribution{weights.begin(), weights.end()};
    }();

    const auto categoryIdDraws =
        views::iota(0u, numActingAgents)
        | views::transform([&](auto) { return multinomial(rng); });

    const auto actorIdsNonCanon =
        categoryIdDraws
        | views::transform([&](auto draw) {
            return std::uniform_int_distribution<uint32_t>{
                0, agentBaseNamesToCounts.at(categoryIdToAgentType.left.at(draw)) - 1}(rng);
        });

    for (auto [categoryId, actorId] : views::zip(categoryIdDraws, actorIdsNonCanon)) {
        if (categoryIdToAgentType.right.at(m_baseName) != categoryId) continue;
        if (!name().ends_with(fmt::format("_{}", actorId))) continue;
        m_price = spotPrice;
        placeOrderChiarella(bookId);
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
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

void StylizedTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag.at(bookId) = false;
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleTrade(Message::Ptr msg)
{
    const auto payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
    const double tradePrice = taosim::util::decimal2double(payload->trade.price());
    m_tradePrice.at(payload->bookId) = {
        .timestamp = msg->arrival,
        .price = tradePrice
    };
}

//-------------------------------------------------------------------------

StylizedTraderAgent::ForecastResult StylizedTraderAgent::forecast(BookId bookId)
{
    const double pf = getProcessValue(bookId, "fundamental");

    m_price = m_priceHist.at(bookId).back();
    const auto& logReturns = m_logReturns.at(bookId);
    const double compF =  1.0 / m_tauF.at(bookId) * std::log(pf/ m_price);
    const double compC = 1.0 / m_historySize * ranges::accumulate(logReturns, 0.0);
    const double compN = std::normal_distribution{0.0, m_sigmaEps}(*m_rng);
    const double tauFNormalizer = m_regimeState.at(bookId) == RegimeState::REGIME_A ?  m_tauF.at(bookId) / m_weightNormalizer *0.01 : 1;
    const double logReturnForecast = m_weightNormalizer
        * (m_weight.F * compF + m_weight.C * compC + m_weight.N * compN) * tauFNormalizer;
    const double varLastLogs = [&] {
            namespace bacc = boost::accumulators;
            bacc::accumulator_set<double, bacc::stats<bacc::tag::lazy_variance>> acc;
            const auto n = logReturns.capacity();
            for (auto logRet : logReturns) {
                acc(logRet);
            }
            return bacc::variance(acc) * (n - 1) / n;
        }();
    
   
    return {
        .price = m_price * std::exp(logReturnForecast*std::log(m_historySize/(m_tauHist/10))),
        .varianceOfLastLogReturns =  varLastLogs};
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::placeOrderChiarella(BookId bookId)
{
    const ForecastResult forecastResult = forecast(bookId);
    

    if (m_riskAversion * forecastResult.varianceOfLastLogReturns == 0.0) {
        return;
    }


    const auto freeBase =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).base.getFree())*0.99;
    const auto freeQuote =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).quote.getFree())*0.99;
    const auto [indifferencePrice, indifferencePriceConverged] =
        calculateIndifferencePrice(forecastResult, freeBase, freeQuote);
    if (!indifferencePriceConverged) return;

    auto [minimumPrice, minimumPriceConverged] =
        calculateMinimumPrice(forecastResult, freeBase, freeQuote);
    if (!minimumPriceConverged) return;

    const auto maximumPrice = forecastResult.price;
    
    if (minimumPrice <= 0.0
        || minimumPrice > indifferencePrice
        || indifferencePrice > maximumPrice) {
        return;
    }

    const double sampledPrice = std::uniform_real_distribution{std::max(minimumPrice,m_priceIncrement), maximumPrice}(*m_rng);

    if (sampledPrice < indifferencePrice) {
        placeLimitBuy(bookId, forecastResult, sampledPrice, freeBase, freeQuote);
    }
    else if (sampledPrice > indifferencePrice) {
        placeLimitSell(bookId, forecastResult, sampledPrice, freeBase, freeQuote);
    }
}

//-------------------------------------------------------------------------

StylizedTraderAgent::OptimizationResult StylizedTraderAgent::calculateIndifferencePrice(
    const StylizedTraderAgent::ForecastResult& forecastResult, double freeBase, double freeQuote)
{
    struct Functor
    {
        ForecastResult forecastResult;
        double riskAversion;
        double freeBase;
        double freeQuote;
        double hara;

        Functor(ForecastResult forecastResult, double riskAversion, double freeBase, double freeQuote, double hara) noexcept
            : forecastResult{forecastResult}, riskAversion{riskAversion}, freeBase{freeBase}, freeQuote{freeQuote},
            hara{hara}
        {}

        int inputs() const noexcept { return 1; }
        int values() const noexcept { return 1; }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            fvec[0] = investmentPosition(x[0], 
                        forecastResult.price, 
                        forecastResult.varianceOfLastLogReturns, 
                        freeBase,
                        freeQuote,
                        riskAversion,
                        hara) 
                    - freeBase;
            return 0;
        }
    };

    Functor functor{forecastResult, m_riskAversion, freeBase, freeQuote, m_hara};
    Eigen::HybridNonLinearSolver<Functor> solver{functor};
    solver.parameters.xtol = 1.49012e-8;
    Eigen::VectorXd x{1};
    x[0] = 1.0;
    Eigen::HybridNonLinearSolverSpace::Status status = solver.hybrd1(x);
    return {
        .value = x[0],
        .converged = status == Eigen::HybridNonLinearSolverSpace::RelativeErrorTooSmall
    };
}

//-------------------------------------------------------------------------

StylizedTraderAgent::OptimizationResult StylizedTraderAgent::calculateMinimumPrice(
    const StylizedTraderAgent::ForecastResult& forecastResult, double freeBase, double freeQuote)
{
    struct Functor
    {
        ForecastResult forecastResult;
        double riskAversion;
        double freeBase;
        double freeQuote;
        double hara;

        Functor(
            ForecastResult forecastResult,
            double riskAversion,
            double freeBase,
            double freeQuote,
            double hara) noexcept
            : forecastResult{forecastResult},
              riskAversion{riskAversion},
              freeBase{freeBase},
              freeQuote{freeQuote},
              hara{hara}
        {}

        int inputs() const noexcept { return 1; }
        int values() const noexcept { return 1; }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
        {
            fvec[0] = x[0] * 
                    (investmentPosition(x[0], 
                            forecastResult.price, 
                            forecastResult.varianceOfLastLogReturns,
                            freeBase,
                            freeQuote,
                            riskAversion,
                            hara) 
                    - freeBase) - freeQuote;
            return 0;
        }
    };

    Functor functor{forecastResult, m_riskAversion, freeBase, freeQuote, m_hara};
    Eigen::HybridNonLinearSolver<Functor> solver{functor};
    solver.parameters.xtol = 1.49012e-8;
    Eigen::VectorXd x{1};
    x[0] = 1.0;
    Eigen::HybridNonLinearSolverSpace::Status status = solver.hybrd1(x);
    return {
        .value = x[0],
        .converged = status == Eigen::HybridNonLinearSolverSpace::RelativeErrorTooSmall
    };
}

// -------------------------------------------------------------------------
double StylizedTraderAgent::calcPositionPrice(const StylizedTraderAgent::ForecastResult& forecastResult, double price, double freeBase, double freeQuote) {
    return investmentPosition(price, forecastResult.price, forecastResult.varianceOfLastLogReturns, freeBase, freeQuote, m_riskAversion, m_hara);
}
//-------------------------------------------------------------------------

void StylizedTraderAgent::placeLimitBuy(
    BookId bookId,
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double sampledPrice,
    double freeBase,
    double freeQuote)
{
    const double price = std::round(sampledPrice / m_priceIncrement) * m_priceIncrement;

    double volume = calcPositionPrice(forecastResult,sampledPrice,freeBase,freeQuote) - freeBase;
    if (const auto attainableVolume = freeQuote / price; volume > attainableVolume) {
        volume = attainableVolume;
    }
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume <= 0.0) {
        return;
    }

    m_orderFlag.at(bookId) = true;

    const float postOnlyProb = std::max(1.0/(1.0 + std::exp(-m_slopeVolGuard* (forecastResult.varianceOfLastLogReturns - m_volGuardX0))), m_alpha);
    const bool postOnly = std::bernoulli_distribution{postOnlyProb}(*m_rng);
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
            bookId,
            Currency::BASE, //#
            std::nullopt,
            postOnly));
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::placeLimitSell(
    BookId bookId,
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double sampledPrice,
    double freeBase,
    double freeQuote)
{
    const double price = std::round(sampledPrice / m_priceIncrement) * m_priceIncrement;

    double volume = freeBase - calcPositionPrice(forecastResult, sampledPrice,freeBase,freeQuote);
    if (volume > freeBase) {
        volume = freeBase;
    }
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume <= 0.0) {
        return;
    }

    m_orderFlag.at(bookId) = true;
    const float postOnlyProb = std::max(1.0/(1.0 + std::exp(-m_slopeVolGuard* (forecastResult.varianceOfLastLogReturns - m_volGuardX0))), m_alpha);
    const bool postOnly = std::bernoulli_distribution{postOnlyProb}(*m_rng);
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
            bookId,
            Currency::BASE, //#
            std::nullopt,
            postOnly));
}

//-------------------------------------------------------------------------

Timestamp StylizedTraderAgent::orderPlacementLatency() {
    return static_cast<Timestamp>(std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

double StylizedTraderAgent::getProcessValue(BookId bookId, const std::string& name)
{
    return simulation()->exchange()->process(name, bookId)->value();
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::updateRegime(BookId bookId)
{
    if (!m_regimeChangeFlag) return;
    
    auto rng = std::mt19937{simulation()->currentTimestamp()};
    if (m_regimeState.at(bookId) == RegimeState::NORMAL
            && std::bernoulli_distribution{m_regimeChangeProb.at(bookId)}(rng)){
        m_tauF.at(bookId) = m_tauFRegime;
        m_regimeState.at(bookId) = RegimeState::REGIME_A;
    } 
    else if (m_regimeState.at(bookId) == RegimeState::REGIME_A 
                && std::bernoulli_distribution{1- std::sqrt(m_regimeChangeProb.at(bookId))}(rng)) {
        m_tauF.at(bookId) = m_tauFOrig;
        m_regimeState.at(bookId) = RegimeState::NORMAL;
    }
}

//------------------------------------------------------------------------- 