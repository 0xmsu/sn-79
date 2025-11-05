/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Simulation.hpp>
#include "taosim/ipc/ipc.hpp"
#include "taosim/replay/ReplayManager.hpp"
#include "net.hpp"

#include <boost/asio.hpp>
#include <pugixml.hpp>

#include <memory>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

struct SimulationBlockInfo
{
    uint32_t count;
    uint32_t dimension;
};

//-------------------------------------------------------------------------

struct NetworkingInfo
{
    std::string host, port, bookStateEndpoint, generalMsgEndpoint;
    int64_t resolveTimeout, connectTimeout, writeTimeout, readTimeout;
};

//-------------------------------------------------------------------------

class SimulationManager
{
public:
    void runSimulations();
    void runReplay();
    void runReplayAdvanced();
    void publishStartInfo();
    void publishEndInfo();
    void publishState();
    void publishStateMessagePack();
    
    [[nodiscard]] SimulationBlockInfo blockInfo() const noexcept { return m_blockInfo; }
    [[nodiscard]] std::span<const std::unique_ptr<Simulation>> simulations() const noexcept { return m_simulations; }
    [[nodiscard]] const fs::path& logDir() const noexcept { return m_logDir; }

    [[nodiscard]] rapidjson::Document makeStateJson() const;
    [[nodiscard]] rapidjson::Document makeCollectiveBookStateJson() const;

    [[nodiscard]] bool online() const noexcept
    {
        return !m_disallowPublish && !m_netInfo.host.empty() && !m_netInfo.port.empty();
    }

    static std::unique_ptr<SimulationManager> fromConfig(const fs::path& path);
    static std::unique_ptr<SimulationManager> fromReplay(const replay::ReplayDesc& desc);

    // TODO: ENV?
    static constexpr std::string_view s_validatorReqMessageQueueName{"taosim-req"};
    static constexpr std::string_view s_validatorResMessageQueueName{"taosim-res"};
    static constexpr std::string_view s_statePublishShmName{"state"};
    static constexpr std::string_view s_remoteResponsesShmName{"responses"};

private:
    SimulationBlockInfo m_blockInfo;
    boost::asio::io_context m_io;
    std::unique_ptr<boost::asio::thread_pool> m_threadPool;
    std::vector<std::unique_ptr<Simulation>> m_simulations;
    fs::path m_logDir;
    Timestamp m_gracePeriod;
    NetworkingInfo m_netInfo;
    UnsyncSignal<void()> m_stepSignal;
    std::unique_ptr<ipc::PosixMessageQueue> m_validatorReqMessageQueue;
    std::unique_ptr<ipc::PosixMessageQueue> m_validatorResMessageQueue;
    bool m_disallowPublish{};
    bool m_useMessagePack{};
    std::unique_ptr<replay::ReplayManager> m_replayManager;

    void setupLogDir(pugi::xml_node node);

    net::awaitable<void> asyncSendOverNetwork(
        const rapidjson::Value& reqBody, const std::string& endpoint, rapidjson::Document& resJson);
    http::request<http::string_body> makeHttpRequest(
        const std::string& target, const std::string& body);
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------