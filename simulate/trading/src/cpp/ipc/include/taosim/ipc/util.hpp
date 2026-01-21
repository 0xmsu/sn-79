/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <ctime>

//-------------------------------------------------------------------------

namespace taosim::ipc
{

//-------------------------------------------------------------------------

[[nodiscard]] inline timespec makeTimespec(size_t ns)
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    static constexpr decltype(ts.tv_nsec) nanos = 1'000'000'000;
    ts.tv_sec += ns / nanos;
    ts.tv_nsec += ns % nanos;
    if (ts.tv_nsec >= nanos) {
        ++ts.tv_sec;
        ts.tv_nsec -= nanos;
    }
    return ts;
}

//-------------------------------------------------------------------------

}  // namespace taosim::ipc

//-------------------------------------------------------------------------
