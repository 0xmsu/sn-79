/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/L3EventLogger.hpp>

#include "Simulation.hpp"
#include "util.hpp"

#include <fmt/chrono.h>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

L3EventLogger::L3EventLogger(
    const fs::path& filepath,
    std::chrono::system_clock::time_point startTimePoint,
    decltype(exchange::ExchangeSignals::L3)& signal,
    Simulation* simulation) noexcept
    : logging::RotatingLoggerBase(logging::RotatingLoggerBaseDesc{
        .name = "L3Logger",
        .simulation = simulation,
        .filepath = filepath,
        .startTimePoint = startTimePoint,
        .header = std::string{s_header}
      }),
      m_feed{signal.connect([this](taosim::L3LogEvent event) { log(event); })}
{}

//-------------------------------------------------------------------------

void L3EventLogger::log(taosim::L3LogEvent event)
{
    updateSink();

    const auto time = m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp());

    rapidjson::Document json = std::visit(
        [&](auto&& item) {
            using T = std::remove_cvref_t<decltype(item)>;
            static_assert(taosim::json::IsL3Serializable<T>);
            rapidjson::Document json;
            item.L3Serialize(json);
            if constexpr (!std::same_as<T, taosim::InstructionLogContext>) {
                json["g"]["b"].SetUint(
                    m_simulation->bookIdCanon(json["g"]["b"].GetUint()));
            }
            json.AddMember("k", rapidjson::Value{event.id}, json.GetAllocator());
            return json;
        },
        event.item);

    m_logger->trace(fmt::format("{:%Y-%m-%d,%H:%M:%S},{}", time, taosim::json::json2str(json)));
    m_logger->flush();
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------