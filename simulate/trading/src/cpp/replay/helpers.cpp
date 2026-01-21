/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/replay/helpers.hpp>

#include <boost/algorithm/string.hpp>
#include <rapidcsv.h>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::replay::helpers
{

//-------------------------------------------------------------------------

fs::path cleanReplayPath(const std::string& path)
{
    const auto pathStr = fs::absolute(path).string();
    if (pathStr.ends_with(fs::path::preferred_separator)) {
        return pathStr.substr(0, pathStr.size() - 1);
    }
    return pathStr;
}

//-------------------------------------------------------------------------

std::unordered_map<Timestamp, decimal_t>
    makeTimestampToMidPriceMapping(const fs::path& L2LogPath)
{
    rapidcsv::Document csv{L2LogPath};

    const auto timestamps = csv.GetColumn<Timestamp>(
        "Time",
        [](const std::string& entry, Timestamp& val) {
            std::istringstream iss{entry};
            std::chrono::nanoseconds t{};
            std::chrono::from_stream(iss, "%T", t);
            val = t.count();
        });
    
    auto emptyHandler = [](const std::string& entry, double& val) {
        val = entry.empty() ? 0.0 : std::stod(entry);
    };
    const auto bidPrices = csv.GetColumn<double>("BidPrice", emptyHandler);
    const auto askPrices = csv.GetColumn<double>("AskPrice", emptyHandler);
    auto midPrices = views::zip(bidPrices, askPrices)
        | views::transform([](auto&& bidAskPair) -> decimal_t {
            const auto bid = util::double2decimal(bidAskPair.first);
            const auto ask = util::double2decimal(bidAskPair.second);
            if (bid == 0_dec || ask == 0_dec) return {};
            return DEC(0.5) * (bid + ask);
        });

    return views::zip(timestamps, midPrices)
        | ranges::to<std::unordered_map<Timestamp, decimal_t>>;
}

//-------------------------------------------------------------------------

Message::Ptr createMessageFromLogFileEntry(const std::string& entry, size_t lineCounter)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    const auto jsonEntryStr = entry.substr(
        boost::algorithm::find_nth(entry, ",", 1).begin() - entry.begin() + 1);
    rapidjson::Document json;
    if (json.Parse(jsonEntryStr.c_str()).HasParseError()) {
        throw std::runtime_error{fmt::format(
            "{}: Error parsing log file entry at line {}: {}", ctx, lineCounter, entry)};
    }

    const std::string msgType{json["p"].GetString()};

    return Message::create(
        json["o"].GetUint64(),
        json["o"].GetUint64() + json["d"].GetUint64(),
        json["s"].GetString(),
        json["t"].GetString(),
        msgType,
        msgType.starts_with("DISTRIBUTED")
            ? MessagePayload::create<DistributedAgentResponsePayload>(
                json["pld"]["a"].GetInt(),
                makePayload(json))
            : makePayload(json));
}

//-------------------------------------------------------------------------

MessagePayload::Ptr makePayload(const rapidjson::Value& json)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    const std::string msgType{json["p"].GetString()};

    const auto& payloadJson =
        msgType.starts_with("DISTRIBUTED") ? json["pld"]["pld"] : json["pld"];

    if (msgType.ends_with("PLACE_ORDER_MARKET")) {
        return MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection{payloadJson["d"].GetUint()},
            json::getDecimal(payloadJson["v"]),
            json::getDecimal(payloadJson["l"]),
            payloadJson["b"].GetUint(),
            Currency{payloadJson["n"].GetUint()},
            payloadJson["ci"].IsNull()
                ? std::nullopt : std::make_optional(payloadJson["ci"].GetUint()),
            magic_enum::enum_cast<taosim::STPFlag>(
                std::string{payloadJson["s"].GetString()}).value(),
            [&] -> taosim::SettleFlag {
                if (payloadJson["f"].IsUint()) {
                    return payloadJson["f"].GetUint();
                } else if (payloadJson["f"].IsString()) {
                    return magic_enum::enum_cast<taosim::SettleType>(
                        std::string{payloadJson["f"].GetString()}).value();
                } else {
                    throw std::runtime_error{fmt::format(
                        "{}: Unrecogized 'settleFlag': {}", ctx, json::json2str(payloadJson["f"]))};
                }
            }());
    }
    else if (msgType.ends_with("PLACE_ORDER_LIMIT")) {
        return MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection{payloadJson["d"].GetUint()},
            json::getDecimal(payloadJson["v"]),
            json::getDecimal(payloadJson["p"]),
            json::getDecimal(payloadJson["l"]),
            payloadJson["b"].GetUint(),
            Currency{payloadJson["n"].GetUint()},
            payloadJson["ci"].IsNull()
                ? std::nullopt : std::make_optional(payloadJson["ci"].GetUint()),
            payloadJson["y"].GetBool(),
            magic_enum::enum_cast<taosim::TimeInForce>(
                std::string{payloadJson["r"].GetString()}).value(),
            payloadJson["x"].IsNull()
                ? std::nullopt : std::make_optional<Timestamp>(payloadJson["x"].GetUint64()),
            magic_enum::enum_cast<taosim::STPFlag>(
                std::string{payloadJson["s"].GetString()}).value(),
            [&] -> taosim::SettleFlag {
                if (payloadJson["f"].IsUint()) {
                    return payloadJson["f"].GetUint();
                } else if (payloadJson["f"].IsString()) {
                    return magic_enum::enum_cast<taosim::SettleType>(
                        std::string{payloadJson["f"].GetString()}).value();
                } else {
                    throw std::runtime_error{fmt::format(
                        "{}: Unrecogized 'settleFlag': {}", ctx, json::json2str(payloadJson["f"]))};
                }
            }());
    }
    else if (msgType.ends_with("CANCEL_ORDERS")) {
        return MessagePayload::create<CancelOrdersPayload>(
            [&] {
                std::vector<event::Cancellation> cancellations;
                for (const auto& cancellationJson : payloadJson["cs"].GetArray()) {
                    cancellations.emplace_back(
                        cancellationJson["i"].GetUint(),
                        cancellationJson["v"].IsNull()
                            ? std::nullopt : std::make_optional(json::getDecimal(cancellationJson["v"])));
                }
                return cancellations;
            }(),
            payloadJson["b"].GetUint());
    }
    else if (msgType.ends_with("CLOSE_POSITIONS")) {
        return MessagePayload::create<ClosePositionsPayload>(
            [&] {
                std::vector<ClosePosition> closePositions;
                for (const auto& closePositionJson : payloadJson["cps"].GetArray()) {
                    closePositions.emplace_back(
                        closePositionJson["i"].GetUint(),
                        closePositionJson["v"].IsNull()
                            ? std::nullopt : std::make_optional(json::getDecimal(closePositionJson["v"])));
                }
                return closePositions;
            }(),
            payloadJson["b"].GetUint());
    }
    else if (msgType.ends_with("RESET_AGENT")) {
        return MessagePayload::create<ResetAgentsPayload>(
            [&] {
                std::vector<AgentId> agentIds;
                for (const auto& agentIdJson : payloadJson["as"].GetArray()) {
                    agentIds.push_back(agentIdJson.GetInt());
                }
                return agentIds;
            }());
    }

    throw std::runtime_error{fmt::format(
        "{}: Unexpected message type encountered during replay: {}", ctx, msgType)};
};

//-------------------------------------------------------------------------

}  // namespace taosim::replay::helpers

//-------------------------------------------------------------------------
