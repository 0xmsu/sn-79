/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "CheckpointSerializable.hpp"
#include <taosim/process/Process.hpp>
#include "common.hpp"

#include <spdlog/spdlog.h>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class BookProcessLogger : public CheckpointSerializable
{
public:
    BookProcessLogger(
        const fs::path& filepath, std::span<const double> X0s, Simulation* simulation);

    [[nodiscard]] const fs::path& filepath() const noexcept { return m_filepath; }

    void log(
        const std::map<BookId, std::vector<double>>& entries, std::span<Timestamp> timestamps);

    virtual void checkpointSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static std::unique_ptr<BookProcessLogger> fromCheckpoint(
        const rapidjson::Value& json, Simulation* simulation);

private:
    BookProcessLogger(const fs::path& filepath);

    fs::path m_filepath;
    std::unique_ptr<spdlog::logger> m_logger;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
