/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/SimulatorState.hpp>
#include <taosim/checkpoint/serialization/helpers.hpp>
#include <taosim/message/serialization/MessageQueue.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct pack<taosim::checkpoint::SimulatorState>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::checkpoint::SimulatorState& v) const
    {
        using namespace taosim::checkpoint::serialization;

        o.pack_map(3);

        o.pack("timestamp");
        o.pack(v.mngr->simulations().front()->currentTimestamp());

        o.pack("blocks");
        o.pack_array(v.mngr->simulations().size());
        for (auto&& simulation : v.mngr->simulations()) {
            o.pack_map(3);

            o.pack("agents");
            packAgents(o, *simulation);

            o.pack("exchange");
            packExchange(o, *simulation);

            o.pack("messageQueue");
            o.pack(simulation->messageQueue());
        }

        o.pack("logFileSizes");
        packLogFileSizes(o, *v.mngr);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
