# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import multiprocessing

def get_core_allocation():
    """
    Allocate CPU cores across validator components using percentage-based allocation.
    """
    total_cores = multiprocessing.cpu_count()
    validator_pct = 0.20    # Main validator loop
    query_pct = 0.20        # Miner queries (subprocess)
    reward_pct = 0.30       # Reward computation (ProcessPool)
    reporting_pct = 0.12    # Metrics reporting (subprocess)
    ipc_pct = 0.10          # IPC operations (ThreadPool)
    
    validator_count = max(2, int(total_cores * validator_pct))
    query_count = max(2, int(total_cores * query_pct))
    reward_count = max(2, int(total_cores * reward_pct))
    reporting_count = max(1, int(total_cores * reporting_pct))
    ipc_count = max(1, int(total_cores * ipc_pct))

    allocated = validator_count + query_count + reward_count + reporting_count + ipc_count
    if allocated > total_cores:
        scale = total_cores / allocated
        validator_count = max(2, int(validator_count * scale))
        query_count = max(2, int(query_count * scale))
        reward_count = max(2, int(reward_count * scale))
        reporting_count = max(1, int(reporting_count * scale))
        ipc_count = max(1, int(ipc_count * scale))
        allocated = validator_count + query_count + reward_count + reporting_count + ipc_count

    offset = 0
    
    validator_cores = list(range(offset, offset + validator_count))
    offset += validator_count
    
    query_cores = list(range(offset, offset + query_count))
    offset += query_count
    
    reward_cores = list(range(offset, offset + reward_count))
    offset += reward_count
    
    reporting_cores = list(range(offset, min(total_cores, offset + reporting_count)))
    offset = min(total_cores, offset + reporting_count)
    
    ipc_cores = list(range(offset, min(total_cores, offset + ipc_count)))

    return {
        'validator': validator_cores,
        'query': query_cores,
        'reward': reward_cores,
        'reporting': reporting_cores,
        'ipc': ipc_cores
    }