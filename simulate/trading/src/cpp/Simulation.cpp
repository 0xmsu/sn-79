/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Simulation.hpp"

#include "taosim/simulation/SimulationException.hpp"
#include "util.hpp"

#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <date/date.h>
#include <date/tz.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <source_location>
#include <stdexcept>

//-------------------------------------------------------------------------

Simulation::Simulation() noexcept
    : IMessageable{this, "SIMULATION"},
      m_localAgentManager{std::make_unique<LocalAgentManager>(this)}
{}

//-------------------------------------------------------------------------

Simulation::Simulation(
    uint32_t blockIdx,
    uint32_t blockDim,
    const fs::path& baseLogDir,
    bool replayMode,
    taosim::replay::ReplayDesc replayDesc)
    : IMessageable{this, "SIMULATION"},
      m_blockIdx{blockIdx},
      m_blockDim{blockDim},
      m_baseLogDir{baseLogDir},
      m_localAgentManager{std::make_unique<LocalAgentManager>(this)},
      m_replayMode{replayMode},
      m_replayDesc{replayDesc}
{}

//-------------------------------------------------------------------------

void Simulation::dispatchMessage(
    Timestamp occurrence,
    Timestamp delay,
    const std::string& source,
    const std::string& target,
    const std::string& type,
    MessagePayload::Ptr payload) const
{
    queueMessage(
        Message::create(occurrence, occurrence + delay, source, target, type, payload));
}

//-------------------------------------------------------------------------

void Simulation::dispatchGenericMessage(
    Timestamp occurrence,
    Timestamp delay,
    const std::string& source,
    const std::string& target,
    const std::string& type,
    std::map<std::string, std::string> payload) const
{
    queueMessage(Message::create(
        occurrence,
        occurrence + delay,
        source,
        target,
        type,
        MessagePayload::create<GenericPayload>(std::move(payload))));
}

//-------------------------------------------------------------------------

void Simulation::queueMessage(Message::Ptr msg) const
{
    m_messageQueue.push(msg);
}

//-------------------------------------------------------------------------

void Simulation::simulate()
{
    if (m_state == taosim::simulation::SimulationState::STOPPED) return;
    else if (m_state == taosim::simulation::SimulationState::INACTIVE) start();

    while (m_time.current < m_time.start + m_time.duration) {
        step();
        m_exchange->L3Record().clear();
    }

    stop();
}

//-------------------------------------------------------------------------

taosim::accounting::Account& Simulation::account(const LocalAgentId& id) const noexcept
{
    return m_exchange->account(id);
}

//-------------------------------------------------------------------------

std::span<const std::unique_ptr<Agent>> Simulation::agents() const noexcept
{
    return m_localAgentManager->agents();
}

//-------------------------------------------------------------------------

Timestamp Simulation::currentTimestamp() const noexcept
{
    return m_time.current;
}

//-------------------------------------------------------------------------

Timestamp Simulation::duration() const noexcept
{
    return m_time.duration;
}

//-------------------------------------------------------------------------

taosim::simulation::SimulationSignals& Simulation::signals() const noexcept
{
    return m_signals;
}

//-------------------------------------------------------------------------

std::mt19937& Simulation::rng() const noexcept
{
    return m_rng;
};

//-------------------------------------------------------------------------

void Simulation::receiveMessage(Message::Ptr msg)
{
    // TODO: Do something?
}

//-------------------------------------------------------------------------

void Simulation::configure(const pugi::xml_node& node)
{
    m_config2 = taosim::simulation::SimulationConfig::fromXML(node);

    pugi::xml_attribute attr;

    if (attr = node.attribute("start"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'start'",
            std::source_location::current().function_name()));
    }
    m_time.start = attr.as_ullong();

    if (attr = node.attribute("duration"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'duration'",
            std::source_location::current().function_name()));
    }
    m_time.duration = attr.as_ullong();

    m_time.step = node.attribute("step").as_ullong(1);
    m_time.current = node.attribute("current").as_ullong(m_time.start);

    if (attr = node.attribute("seed"); !attr.empty()) {
        m_rng = std::mt19937{attr.as_ullong()};
    } else {
        m_rng = std::mt19937{std::random_device{}()};
    }

    m_config = [node] -> std::string {
        std::ostringstream oss;
        node.print(oss);
        return oss.str();
    }();

    if (node.attribute("debug").as_bool()) {
        m_debug = true;
    }

    if (node.attribute("error").as_bool()) {
        m_error = true;
    }

    m_logWindow = [&] -> Timestamp {
        static constexpr const char* attrName = "logWindow";
        const auto logWindow = node.attribute(attrName).as_ullong();
        using namespace taosim::simulation;
        if (!(logWindow == 0ul || kLogWindowMin <= logWindow && logWindow <= kLogWindowMax)) {
            throw std::runtime_error{fmt::format(
                "{}: '{}' must be either empty (or 0), or in [{}, {}], was {}",
                std::source_location::current().function_name(),
                attrName,
                kLogWindowMin, kLogWindowMax,
                node.attribute(attrName).as_string())};
        }
        return logWindow;
    }();

    // NOTE: Ordering important!
    configureLogging(node);
    configureAgents(node);
}

//-------------------------------------------------------------------------

void Simulation::saveCheckpoint()
{
    fmt::println("Saving checkpoint...");
    rapidjson::Document json{rapidjson::kObjectType};
    auto& allocator = json.GetAllocator();

    // Config file contents, with current id and time added.
    logDebug("Serializing config...");
    taosim::json::serializeHelper(
        json,
        "config",
        [this](rapidjson::Document& json) {
            const auto config = [this] -> std::string {
                pugi::xml_document doc;
                pugi::xml_parse_result result = doc.load_string(m_config.c_str());
                pugi::xml_node node = doc.child("Simulation");
                if (pugi::xml_attribute attr = node.attribute("id")) {
                    attr.set_value(m_id.c_str());
                } else {
                    node.append_attribute("id") = m_id.c_str();
                }
                if (pugi::xml_attribute attr = node.attribute("current")) {
                    attr.set_value(m_time.current);
                } else {
                    node.append_attribute("current") = m_time.current;
                }
                std::ostringstream oss;
                doc.print(oss, "");
                return oss.str();
            }();
            json.SetString(config.c_str(), config.size(), json.GetAllocator());
        });

    // Accounts, including standing orders.
    logDebug("Serializing accounts...");
    m_exchange->checkpointSerialize(json, "accounts");

    // Active OrderContexts.
    logDebug("Serializing OrderContexts...");
    taosim::json::serializeHelper(
        json,
        "order2clientCtx",
        [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto& book : m_exchange->books()) {
                rapidjson::Document subJson{rapidjson::kObjectType, &allocator};
                for (const auto& [orderId, clientCtx] : book->m_order2clientCtx) {
                    clientCtx.checkpointSerialize(subJson, std::to_string(orderId));
                }
                json.PushBack(subJson, allocator);
            }
        }
    );

    // ID counters of order & trade factories.
    logDebug("Serializing Order & Trade factories...");
    taosim::json::serializeHelper(
        json,
        "orderIdCounters",
        [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto& book : m_exchange->books()) {
                rapidjson::Document subJson{&allocator};
                book->orderFactory().checkpointSerialize(subJson);
                json.PushBack(subJson, allocator);
            }
        });
    taosim::json::serializeHelper(
        json,
        "tradeIdCounters",
        [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto& book : m_exchange->books()) {
                rapidjson::Document subJson{&allocator};
                book->tradeFactory().checkpointSerialize(subJson);
                json.PushBack(subJson, allocator);
            }
        });

    // Subscriptions.
    logDebug("Serializing Subscriptions...");
    taosim::json::serializeHelper(
        json,
        "subscriptions",
        [this](rapidjson::Document& json) {
            json.SetObject();
            m_exchange->m_localMarketOrderSubscribers.checkpointSerialize(json, "market");
            logDebug("Serialized localMarketOrderSubscribers");
            m_exchange->m_localLimitOrderSubscribers.checkpointSerialize(json, "limit");
            logDebug("Serialized localLimitOrderSubscribers");
            m_exchange->m_localTradeSubscribers.checkpointSerialize(json, "trade");
            logDebug("Serialized localTradeSubscribers");
            taosim::json::serializeHelper(
                json,
                "tradeByOrder",
                [this](rapidjson::Document& json) {
                    json.SetObject();
                    for (const auto& [orderId, subs] : m_exchange->m_localTradeByOrderSubscribers) {
                        subs.checkpointSerialize(json, std::to_string(orderId));
                    }
                });
        });

    // Misc.
    logDebug("Serializing Misc...");
    json.AddMember("retainRecord", rapidjson::Value{m_exchange->m_retainRecord}, allocator);
    json.AddMember(
        "checkpointWriteTime",
        rapidjson::Value{
            date::format(
                "%Y%m%d_%H%M%S",
                date::make_zoned(
                    date::current_zone(),
                    std::chrono::system_clock::now())).c_str(), allocator},
        allocator);
    json.AddMember("state", rapidjson::Value{std::to_underlying(m_state)}, allocator);
    m_exchange->m_bookProcessManager->checkpointSerialize(json, "processManager");    

    // Save to disk...
    const fs::path ckptDir = m_logDir / "ckpt";
    logDebug("Replacing checkpoint data at {}...", ckptDir.string());
    fs::create_directories(ckptDir);

    // ...and the log files up to this point.
    logDebug("Storing log file sizes...");
    taosim::json::serializeHelper(
        json,
        "logs",
        [this](rapidjson::Document& json) -> void {
            json.SetObject();
            auto& allocator = json.GetAllocator();
            for ([[maybe_unused]] const auto& [_, logger] : m_exchange->m_L2Loggers) {
                // std::string lastLine = util::getLastLines(logger->filepath().string(), 1)[0];
                std::ifstream source{logger->filepath().generic_string(), std::ios_base::binary};
                source.seekg(0, std::ios_base::end);
                const size_t size = static_cast<size_t>(source.tellg());
                logDebug("Storing size {} for file {}...", size, logger->filepath().c_str());
                json.AddMember(
                    rapidjson::Value{logger->filepath().filename().c_str(), allocator},
                    rapidjson::Value{size},
                    allocator);
            }
            for ([[maybe_unused]] const auto& [_, logger] : m_exchange->m_L3EventLoggers) {
                // std::string lastLine = util::getLastLines(logger->filepath().string(), 1)[0];
                std::ifstream source{logger->filepath().generic_string(), std::ios_base::binary};
                source.seekg(0, std::ios_base::end);
                const size_t size = static_cast<size_t>(source.tellg());
                logDebug("Storing size {} for file {}...", size, logger->filepath().c_str());
                json.AddMember(
                    rapidjson::Value{logger->filepath().filename().c_str(), allocator},
                    rapidjson::Value{size},
                    allocator);
            }
        }
    );

    const fs::path ckptPath = ckptDir / "ckpt.json";
    const fs::path ckptTmpPath = ckptDir / "ckpt.tmp.json";
    std::ofstream ckptTmpFile{ckptTmpPath};
    if (!ckptTmpFile) {
        throw std::runtime_error(fmt::format(
            "{}: Error writing checkpoint to '{}'",
            std::source_location::current().function_name(),
            ckptTmpPath.c_str()));
    }
    logDebug("Writing new checkpoint data...");
    taosim::json::dumpJson(json, ckptTmpFile);
    logDebug("Cleaning up...");
    fs::remove(ckptPath);
    fs::rename(ckptTmpPath, ckptPath);

    fmt::println("Checkpoint saved!");
}

//-------------------------------------------------------------------------

std::unique_ptr<Simulation> Simulation::fromCheckpoint(const fs::path& path)
{
    fmt::println("Resuming simulation from checkpoint at {}", path.string());
    rapidjson::Document json = taosim::json::loadJson(path);

    auto simulation = std::make_unique<Simulation>();

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_string(json["config"].GetString());
    pugi::xml_node node = doc.child("Simulation");

    // Config.
    fmt::println("Configuring simulation...");
    simulation->configure(node);
    
    fmt::println("\nRestoring accounts...");
    simulation->exchange()->accounts().registerJson(json["accounts"]);

    // Books & Accounts.
    fmt::println("Restoring books...");
    int cnt = 0;
    const BookId bookCount = simulation->exchange()->books().size();
    for (const auto& member : json["accounts"].GetObject()) {
        const rapidjson::Value& accountJson = member.value;
        const AgentId agentId = accountJson["agentId"].GetInt();
        for (BookId bookId = 0; bookId < bookCount; ++bookId) {
            Book::Ptr book = simulation->exchange()->books()[bookId];
            book->m_orderFactory.m_idCounter =
                json["orderIdCounters"][bookId]["idCounter"].GetUint64();
            book->m_tradeFactory.m_idCounter =
                json["tradeIdCounters"][bookId]["idCounter"].GetUint64();
            book->m_initMode = true;
            const rapidjson::Value& bookOrdersJson = accountJson["orders"][bookId];
            for (const rapidjson::Value& orderJson : bookOrdersJson.GetArray()) {
                LimitOrder::Ptr order = LimitOrder::fromJson(
                    orderJson, 
                    simulation->exchange()->m_config.parameters().priceIncrementDecimals, 
                    simulation->exchange()->m_config.parameters().volumeIncrementDecimals);
                book->m_order2clientCtx[order->id()] =
                    OrderClientContext::fromJson(
                        json["order2clientCtx"][bookId][std::to_string(order->id()).c_str()]);
                book->placeOrder(order);
                simulation->exchange()->accounts()[agentId].activeOrders()[bookId].insert(order);
                fmt::print("Restored Agent {} Book {} Order {}         \r", agentId, bookId, order->id());
                cnt++;
            }
            book->m_initMode = false;
        }
    }

    // Subscriptions.
    fmt::println("Restoring Subscriptions: localMarketOrderSubscribers");
    simulation->exchange()->m_localMarketOrderSubscribers =
        SubscriptionRegistry<LocalAgentId>::fromJson(json["subscriptions"]["market"]);
    fmt::println("Restoring Subscriptions: localLimitOrderSubscribers");
    simulation->exchange()->m_localLimitOrderSubscribers =
        SubscriptionRegistry<LocalAgentId>::fromJson(json["subscriptions"]["limit"]);
    fmt::println("Restoring Subscriptions: localTradeSubscribers");
    simulation->exchange()->m_localTradeSubscribers = 
        SubscriptionRegistry<LocalAgentId>::fromJson(json["subscriptions"]["trade"]);
    fmt::println("Restoring Subscriptions: localTradeByOrderSubscribers");
    for (const auto& member : json["subscriptions"]["tradeByOrder"].GetObject()) {
        const char* name = member.name.GetString();
        const BookId bookId = std::stoi(name);
        simulation->exchange()->m_localTradeByOrderSubscribers[bookId] =
            SubscriptionRegistry<LocalAgentId>::fromJson(
                json["subscriptions"]["tradeByOrder"][name]);
    }

    // Misc.
    fmt::println("Restoring Misc..");
    simulation->exchange()->retainRecord(json["retainRecord"].GetBool());
    simulation->m_state = taosim::simulation::SimulationState{json["state"].GetUint()};
    simulation->exchange()->m_bookProcessManager = BookProcessManager::fromCheckpoint(
        json["processManager"],
        simulation.get(),
        const_cast<taosim::exchange::ExchangeConfig*>(&simulation->exchange()->config2()));

    // Replace log files with those from the checkpoint.
    fmt::println("Aligning Logs with Checkpoint..");
    std::regex pattern{".*L[23].*\\.log$"};
    for (const fs::directory_entry& entry : fs::directory_iterator(simulation->m_logDir)) {
        const std::string filename = entry.path().filename().string();
        if (fs::is_regular_file(entry) && std::regex_match(filename, pattern)) {
            off_t size = json["logs"][filename.c_str()].GetInt();
            auto logFile = (simulation->m_logDir / filename);
            fs::resize_file(logFile, size);
            fmt::println("Truncated file {} to size {}", logFile.string(), size);
        }
    }
    fmt::println("Resumed from checkpoint!");
    return simulation;
}

//-------------------------------------------------------------------------

void Simulation::configureAgents(pugi::xml_node node)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    static const std::set<std::string> specialAgents{
        "DISTRIBUTED_PROXY_AGENT",
        "EXCHANGE",
        "LOGGER_TRADES"
    };

    pugi::xml_node agentsNode;

    if (agentsNode = node.child("Agents"); !agentsNode) {
        throw std::invalid_argument{fmt::format(
            "{}: missing required child 'Agents'", ctx)};
    }

    m_localAgentManager->createAgentsInstanced(
        agentsNode,
        [&](pugi::xml_node agentNode) {
            if (specialAgents.contains(agentNode.attribute("name").as_string())) return;
            if (m_exchange == nullptr) {
                throw std::runtime_error{fmt::format("{}: m_exchange == nullptr!", ctx)};
            }
            [&] {
                const std::string agentType = agentNode.name();
                auto& accounts = m_exchange->accounts();
                if (m_exchange->accounts().agentTypeAccountTemplates().contains(agentType)) return;
                if (pugi::xml_node balancesNode = agentNode.child("Balances")) {
                    auto doc = std::make_shared<pugi::xml_document>();
                    doc->append_copy(balancesNode);
                    m_exchange->accounts().setAccountTemplate(
                        agentType,
                        [=, this] -> taosim::accounting::Account {
                            const auto& params = m_exchange->config().parameters();
                            return taosim::accounting::Account{
                                static_cast<uint32_t>(m_exchange->books().size()),
                                taosim::accounting::Balances::fromXML(
                                    doc->child("Balances"),
                                    taosim::accounting::RoundParams{
                                        .baseDecimals = params.baseIncrementDecimals,
                                        .quoteDecimals = params.quoteIncrementDecimals
                                    })
                                };
                        });
                }
            }();
            [&] {
                if (pugi::xml_node feePolicyNode = agentNode.child("FeePolicy")) {
                    auto feePolicy = m_exchange->clearingManager().feePolicy();
                    const auto agentBaseName = agentNode.attribute("name").as_string();
                    if (feePolicy->contains(agentBaseName)) return;

                    if (std::string(agentNode.child("FeePolicy").attribute("type").as_string()) == "tiered") {
                        if (!feePolicy->isTiered()){
                            throw std::runtime_error(fmt::format(
                                "{}'s fee policy type must be the same as default", std::string(agentBaseName)));
                        }
                        (*feePolicy)[agentBaseName] =
                            taosim::exchange::TieredFeePolicy::fromXML(feePolicyNode, this);
                        logDebug("TIERED FEE POLICY - {}", agentBaseName);
                        int c = 0;
                        if (auto* tiered = dynamic_cast<TieredFeePolicy*>((*feePolicy)[agentBaseName].get())) {
                            for (auto& tier : tiered->tiers()) {
                                logDebug("TIER {} : VOL >= {} | MAKER {} TAKER {}", c, 
                                    tier.volumeRequired, 
                                    tier.makerFeeRate, tier.takerFeeRate
                                );
                                c++;
                            }
                        }
                    } else {
                        if (feePolicy->isTiered()) {
                            if (auto* tiered = dynamic_cast<TieredFeePolicy*>(feePolicy->defaultPolicy())) {
                                (*feePolicy)[agentBaseName] = std::make_unique<TieredFeePolicy>(*tiered);
                                logDebug("DEFAULT TIERED FEE POLICY - {}", agentBaseName);
                            } else {
                                throw std::runtime_error("Default policy is not TieredFeePolicy as expected");
                            }
                        } 
                    }
                }
            }();
        });

    static constexpr std::array<std::pair<std::string_view, std::string_view>, 2> kSpecialAgents{{
        {"EXCHANGE", "MultiBookExchangeAgent"},
        {"DISTRIBUTED_PROXY_AGENT", "DistributedProxyAgent"}
    }};
    for (const auto& [name, nodeName] : kSpecialAgents) {
        auto it = ranges::find_if(
            *m_localAgentManager, [&](const auto& agent) { return agent->name() == name; });
        if (it == m_localAgentManager->end()) {
            throw std::invalid_argument{fmt::format(
                "{}: missing required agent node '{}'", ctx, nodeName)};
        }
    }

    for (const auto& agent : m_localAgentManager->agents()) {
        if (specialAgents.contains(agent->name())) continue;
        m_exchange->accounts().registerLocal(agent->name(), agent->type());
    }

    m_signals.agentsCreated();
}

//-------------------------------------------------------------------------

void Simulation::configureLogging(pugi::xml_node node)
{
    m_logDir = m_baseLogDir;
}

//-------------------------------------------------------------------------

void Simulation::deliverMessage(Message::Ptr msg)
{
    for (const auto& target : msg->targets) {
        if (target == "*") {
            receiveMessage(msg);
            for (const auto& agent : m_localAgentManager->agents()) {
                agent->receiveMessage(msg);
            }
        }
        else if (target == "EXCHANGE") {
            m_exchange->receiveMessage(msg);
        }
        else if (target == name()) {
            receiveMessage(msg);
        }
        else if (target.back() == '*') {
            const auto prefix = target.substr(0, target.size() - 1);
            auto lb = std::lower_bound(
                m_localAgentManager->begin(),
                m_localAgentManager->end(),
                prefix,
                [](const auto& agent, const auto& needle) {
                    const auto& haystack = agent->name();
                    return haystack.find(needle);
                });
            auto ub = std::upper_bound(
                m_localAgentManager->begin(),
                m_localAgentManager->end(),
                prefix,
                [](const auto& needle, const auto& agent) {
                    const auto& haystack = agent->name();
                    return haystack.find(needle);
                });
            std::for_each(lb, ub, [msg](const auto& agent) { agent->receiveMessage(msg); });
        }
        else {
            auto it = std::lower_bound(
                m_localAgentManager->begin(),
                m_localAgentManager->end(),
                target,
                [](const auto& agent, const auto& val) { return agent->name() < val; });
            if ((*it)->name() != target) {
                return;
            }
            else if (it == m_localAgentManager->end()) {
                throw taosim::simulation::SimulationException(fmt::format(
                    "{}: unknown message target '{}'",
                    std::source_location::current().function_name(),
                    target));
            }
            (*it)->receiveMessage(msg);
        }
    }
}

//-------------------------------------------------------------------------

void Simulation::start()
{
    if (!m_replayMode) {
        dispatchMessage(
            m_time.start,
            0,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_START",
            MessagePayload::create<StartSimulationPayload>(logDir().generic_string()));
        dispatchMessage(
            m_time.start,
            m_time.duration - 1,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_END",
            MessagePayload::create<EmptyPayload>());
    }
    else if (!m_replayDesc.replacedAgents.empty()) {
        auto replacedAgentNames = m_localAgentManager->agents()
            | views::filter([this](auto&& agent) {
                const std::regex pat{fmt::format(
                    "^({})_(\\d+)$", fmt::join(m_replayDesc.replacedAgents, "|"))};
                return std::regex_match(agent->name(), pat);
            })
            | views::transform([](auto&& agent) {
                return agent->name();
            });
        dispatchMessage(
            m_time.start,
            0,
            "SIMULATION",
            fmt::format("{}", fmt::join(replacedAgentNames, "|")),
            "EVENT_SIMULATION_START",
            MessagePayload::create<StartSimulationPayload>(logDir().generic_string()));
        dispatchMessage(
            m_time.start,
            m_time.duration - 1,
            "SIMULATION",
            fmt::format("{}", fmt::join(replacedAgentNames, "|")),
            "EVENT_SIMULATION_END",
            MessagePayload::create<EmptyPayload>());
    }

    m_state = taosim::simulation::SimulationState::STARTED;
    m_signals.start();
}

//-------------------------------------------------------------------------

void Simulation::step()
{
    const Timestamp cutoff = m_time.current + m_time.step;

    m_exchange->clearingManager().updateFeeTiers(cutoff);        
    m_exchange->checkMarginCall();

    auto loopCondition = [&] -> bool {
        return !m_messageQueue.empty()
            && m_messageQueue.top()->arrival < cutoff;
    };

    while (loopCondition()) {
        Message::Ptr msg = m_messageQueue.top();
        m_messageQueue.pop();
        updateTime(msg->arrival);
        deliverMessage(msg);
    }

    updateTime(std::max(m_time.current, cutoff));
    m_signals.step();
}

//-------------------------------------------------------------------------

void Simulation::stop()
{
    m_state = taosim::simulation::SimulationState::STOPPED;
    m_signals.stop();
}

//-------------------------------------------------------------------------

bool Simulation::isReplacedAgent(const std::string& name) const noexcept
{
    auto name2BaseName = [](auto&& name) {
        std::string res = name;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    };
    return m_replayDesc.replacedAgents.contains(name2BaseName(name));
}

//-------------------------------------------------------------------------

bool Simulation::isInitAgent(const std::string& name) const noexcept
{
    auto name2BaseName = [](auto&& name) {
        std::string res = name;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    };
    return name2BaseName(name) == "INITIALIZATION_AGENT";
}

//-------------------------------------------------------------------------
