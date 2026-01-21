/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Simulation.hpp>
#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/ipc/ipc.hpp>
#include <taosim/replay/ReplayManager.hpp>
#include <net.hpp>

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
    
    [[nodiscard]] SimulationBlockInfo blockInfo() const noexcept { return m_blockInfo; }
    [[nodiscard]] auto&& threadPool(this auto&& self) noexcept { return self.m_threadPool; }
    [[nodiscard]] auto&& simulations(this auto&& self) noexcept { return self.m_simulations; }
    [[nodiscard]] const fs::path& logDir() const noexcept { return m_logDir; }
    [[nodiscard]] auto&& stepSignal(this auto&& self) noexcept { return self.m_stepSignal; }
    [[nodiscard]] auto&& checkpointManager(this auto&& self) noexcept { return self.m_checkpointManager; }
    [[nodiscard]] auto&& measurements(this auto&& self) noexcept { return self.m_measurements; }

    [[nodiscard]] bool online() const noexcept;
    [[nodiscard]] bool warmingUp() const noexcept;

    static std::unique_ptr<SimulationManager> fromConfig(const fs::path& configPath, const fs::path& baseDir);
    static std::unique_ptr<SimulationManager> fromCheckpoint(const checkpoint::CheckpointToken& ckptToken);
    static std::unique_ptr<SimulationManager> fromReplay(const replay::ReplayDesc& desc);

    // TODO: ENV?
    static constexpr std::string_view s_validatorReqMessageQueueName{"taosim-req"};
    static constexpr std::string_view s_validatorResMessageQueueName{"taosim-res"};
    static constexpr std::string_view s_statePublishShmName{"state"};
    static constexpr std::string_view s_remoteResponsesShmName{"responses"};

private:
    void setupLogDir(pugi::xml_node simuNode, const fs::path& logPath);
    void publishStateJson();
    void publishStateMessagePack();

    [[nodiscard]] rapidjson::Document makeStateJson() const;
    [[nodiscard]] rapidjson::Document makeCollectiveBookStateJson() const;

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
    bool m_useMessagePack{};
    bool m_replayMode{};
    std::unique_ptr<replay::ReplayManager> m_replayManager;
    std::unique_ptr<checkpoint::CheckpointManager> m_checkpointManager;
    bool m_measureStepWallClockTime{};
    struct {
        using Measurement = decltype(std::chrono::high_resolution_clock::now());
        std::optional<Measurement>
            t0parse, t1parse,
            t0proc, t1proc,
            t0state, t1state,
            t0ckptSave, t1ckptSave,
            t0ckptLoad, t1ckptLoad;
    } m_measurements;

    net::awaitable<void> asyncSendOverNetwork(
        const rapidjson::Value& reqBody, const std::string& endpoint, rapidjson::Document& resJson);
    http::request<http::string_body> makeHttpRequest(
        const std::string& target, const std::string& body);
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------