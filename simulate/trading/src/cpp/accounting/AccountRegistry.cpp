/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/AccountRegistry.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

Account& AccountRegistry::at(const std::variant<AgentId, LocalAgentId>& agentId)
{
    return std::visit(
        [this](auto&& agentId) -> Account& {
            using T = std::remove_cvref_t<decltype(agentId)>;
            if constexpr (std::same_as<T, AgentId>) {
                return m_accounts[agentId];
            } else {
                return m_accounts[m_idBimap.left.at(agentId)];
            }
        },
        agentId);
}

//-------------------------------------------------------------------------

Account& AccountRegistry::operator[](const std::variant<AgentId, LocalAgentId>& agentId)
{
    return at(agentId);
}

//-------------------------------------------------------------------------

void AccountRegistry::registerLocal(
    const LocalAgentId& agentId, std::optional<Account> account) noexcept
{
    const auto id = --m_localIdCounter;
    m_idBimap.insert({agentId, id});
    m_accounts[id] = account.value_or(m_accountTemplate());
    m_agentIdToBaseName[id] = [&] {
        std::string res = agentId;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();
}

//-------------------------------------------------------------------------

void AccountRegistry::registerLocal(
    const LocalAgentId& agentId,
    const std::string& agentType,
    std::optional<Account> account) noexcept
{
    const auto id = --m_localIdCounter;
    m_idBimap.insert({agentId, id});
    m_accounts[id] = account.or_else(
        [&] -> std::optional<taosim::accounting::Account> {
            auto it = m_agentTypeAccountTemplates.find(agentType);
            if (it == m_agentTypeAccountTemplates.end()) return std::nullopt;
            return std::make_optional(it->second());
        })
        .value_or(m_accountTemplate());
    m_agentIdToBaseName[id] = [&] {
        std::string res = agentId;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();
}

//-------------------------------------------------------------------------

AgentId AccountRegistry::registerRemote(std::optional<Account> account) noexcept
{
    m_accounts[m_remoteIdCounter] = account.value_or(m_accountTemplate());
    return m_remoteIdCounter++;
}

//-------------------------------------------------------------------------

void AccountRegistry::registerJson(const rapidjson::Value& json)
{
    for (const auto& member : json.GetObject()) {
        const rapidjson::Value& accountJson = member.value;
        const AgentId agentId = accountJson["agentId"].GetInt();
        if (!accountJson["agentName"].IsNull()) {
            m_idBimap.left.insert({accountJson["agentName"].GetString(), agentId});
        }
        for (const rapidjson::Value& balanceJson : accountJson["balances"].GetArray()) {
            const BookId bookId = balanceJson["bookId"].GetUint();
            m_accounts.at(agentId).at(bookId) = Balances::fromJson(balanceJson);
            fmt::println("AGENT #{} BOOK {} : RESTORED BALANCES : QUOTE {} | BASE {}",
                agentId, bookId,
                m_accounts.at(agentId).at(bookId).quote, m_accounts.at(agentId).at(bookId).base);
        }
        if (agentId < 0) {
            m_localIdCounter = std::min(m_localIdCounter, agentId);
        } else {
            m_remoteIdCounter = std::max(m_remoteIdCounter, agentId + 1);
        }
    }
}

//-------------------------------------------------------------------------

bool AccountRegistry::contains(const std::variant<AgentId, LocalAgentId>& agentId) const
{
    return std::visit(
        [this](auto&& agentId) {
            using T = std::remove_cvref_t<decltype(agentId)>;
            if constexpr (std::same_as<T, AgentId>) {
                return m_accounts.contains(agentId);
            } else {
                return m_accounts.contains(m_idBimap.left.at(agentId));
            }
        },
        agentId);
}

//-------------------------------------------------------------------------

AgentId AccountRegistry::getAgentId(const std::variant<AgentId, LocalAgentId>& agentId) const
{
    return std::visit(
        [this](auto&& agentId) {
            using T = std::remove_cvref_t<decltype(agentId)>;
            if constexpr (std::same_as<T, AgentId>) {
                return agentId;
            } else if constexpr (std::same_as<T, LocalAgentId>) {
                return m_idBimap.left.at(agentId);
            } else {
                static_assert(false, "Non-exhaustive visitor for agentId");
            }
        },
        agentId);
}

//-------------------------------------------------------------------------

std::optional<std::reference_wrapper<const std::string>> AccountRegistry::getAgentBaseName(
    AgentId agentId) const noexcept
{
    auto it = m_agentIdToBaseName.find(agentId);
    if (it == m_agentIdToBaseName.end()) return std::nullopt;
    return std::make_optional(std::cref(it->second));
}

//-------------------------------------------------------------------------

void AccountRegistry::setAccountTemplate(std::function<Account()> factory) noexcept
{
    m_accountTemplate = factory;
}

//-------------------------------------------------------------------------

void AccountRegistry::setAccountTemplate(
    const std::string& agentType, std::function<Account()> factory) noexcept
{
    m_agentTypeAccountTemplates.emplace(agentType, factory);
}

//-------------------------------------------------------------------------

void AccountRegistry::reset(AgentId agentId)
{
    auto baseNameIt = m_agentIdToBaseName.find(agentId);
    if (baseNameIt == m_agentIdToBaseName.end()) {
        m_accounts[agentId] = m_accountTemplate();
        return;
    }
    auto typeTemplateIt = m_agentTypeAccountTemplates.find(baseNameIt->second);
    if (typeTemplateIt == m_agentTypeAccountTemplates.end()) {
        m_accounts[agentId] = m_accountTemplate();
        return;
    }
    m_accounts[agentId] = typeTemplateIt->second();
}

//-------------------------------------------------------------------------

void AccountRegistry::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        for (const auto& [agentId, account] : m_accounts) {
            rapidjson::Document accountJson{rapidjson::kObjectType, &allocator};
            accountJson.AddMember("agentId", rapidjson::Value{agentId}, allocator);
            account.jsonSerialize(accountJson, "balances");
            accountJson.AddMember(
                "agentName",
                agentId < 0
                    ? rapidjson::Value{m_idBimap.right.at(agentId).c_str(), allocator}.Move()
                    : rapidjson::Value{}.SetNull(),
                allocator);
            json.AddMember(
                rapidjson::Value{std::to_string(agentId).c_str(), allocator},
                accountJson,
                allocator);
        }
    };
    json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------