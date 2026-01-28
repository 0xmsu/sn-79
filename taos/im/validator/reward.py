# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import torch
import random
import bittensor as bt
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.utils import normalize
from taos.im.utils.kappa import kappa_3, batch_kappa_3, _get_pnl_fingerprint

def score_uid(validator_data: Dict, uid: int) -> float:
    """
    Calculates the new score value for a specific UID using realized Kappa-3 metric only.

    Args:
        validator_data (Dict): Dictionary containing validator state
        uid (int): UID of miner being scored

    Returns:
        float: The new score value for the given UID.
    """
    kappa_values = validator_data['kappa_values']
    activity_factors = validator_data['activity_factors']
    roundtrip_volumes = validator_data['roundtrip_volumes']  # Full data
    config = validator_data['config']['scoring']
    simulation_config = validator_data['simulation_config']

    simulation_timestamp = validator_data['simulation_timestamp']
    publish_interval = simulation_config['publish_interval']

    if not kappa_values[uid]:
        return 0.0

    uid_kappa = kappa_values[uid]
    kappas = uid_kappa['books']
    
    norm_min = config['kappa']['normalization_min']
    norm_max = config['kappa']['normalization_max']
    norm_range = norm_max - norm_min
    norm_range_inv = 1.0 / norm_range if norm_range != 0 else 0.0
    
    # Normalize kappas per book - treat None as 0.0 (no valid kappa = zero score for that book)
    normalized_kappas = {
        book_id: (max(0.0, min(1.0, (kappa_val - norm_min) * norm_range_inv)) if kappa_val is not None else 0.0)
        for book_id, kappa_val in kappas.items()
    }

    volume_cap = round(
        config['activity']['capital_turnover_cap'] * simulation_config['miner_wealth'],
        simulation_config['volumeDecimals']
    )
    volume_cap_inv = 1.0 / volume_cap

    lookback = config['kappa']['lookback']
    
    decay_grace_period = config['activity'].get('decay_grace_period', 600_000_000_000)
    activity_impact = config['activity'].get('impact', 0.33)
    time_acceleration_power = 2.0
    
    scoring_interval_seconds = config['interval'] / 1e9
    total_intervals = lookback // scoring_interval_seconds
    grace_intervals = decay_grace_period / config['interval']
    decay_window_intervals = total_intervals - grace_intervals
    
    # Safety check for decay window
    if decay_window_intervals <= 0:
        bt.logging.warning(f"UID {uid}: Invalid decay window (total={total_intervals}, grace={grace_intervals})")
        base_decay_factor = 0.999  # Fallback to very slow decay
    else:
        base_decay_factor = 2 ** (-1 / decay_window_intervals)
        base_decay_factor = max(0.5, min(0.9999, base_decay_factor))  # Clamp to safe range
    
    decay_window_ns = (lookback * config['interval']) - decay_grace_period
    decay_window_ns_inv = 1.0 / decay_window_ns if decay_window_ns > 0 else 0.0

    # ===== COMPUTE COMPACT ROUNDTRIP VOLUMES ON-THE-FLY =====
    lookback_threshold = simulation_timestamp - (lookback * publish_interval)
    sampling_interval = config['activity']['trade_volume_sampling_interval']
    sampled_timestamp = (simulation_timestamp // sampling_interval) * sampling_interval
    
    # Get number of books from normalized_kappas keys
    num_books = len(normalized_kappas)
    
    # Compute compact roundtrip volumes for this UID
    miner_roundtrip_volumes = {}
    latest_roundtrip_volumes = {}
    latest_roundtrip_timestamps = {}
    
    if uid in roundtrip_volumes:
        uid_rt_volumes = roundtrip_volumes[uid]
        
        for book_id in range(num_books):
            if book_id in uid_rt_volumes:
                rt_volumes = uid_rt_volumes[book_id]
                
                if rt_volumes:
                    lookback_volume = 0.0
                    latest_time = 0
                    latest_volume = 0.0
                    
                    for ts, vol in rt_volumes.items():
                        if ts >= lookback_threshold:
                            lookback_volume += vol
                        if vol > 0 and ts <= sampled_timestamp and ts > latest_time:
                            latest_time = ts
                    
                    if latest_time > 0 and latest_time >= sampled_timestamp - sampling_interval:
                        latest_volume = rt_volumes[latest_time]
                    
                    miner_roundtrip_volumes[book_id] = lookback_volume
                    latest_roundtrip_volumes[book_id] = latest_volume
                    latest_roundtrip_timestamps[book_id] = latest_time
                else:
                    miner_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_timestamps[book_id] = 0
            else:
                miner_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_timestamps[book_id] = 0
    else:
        # UID not in roundtrip_volumes, initialize all books to zero
        for book_id in range(num_books):
            miner_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_timestamps[book_id] = 0
    # ===== END COMPACT COMPUTATION =====
    
    # Calculate the activity factors to be multiplied onto the Kappas
    activity_factors_uid = activity_factors[uid]
    decay_rate = config['activity'].get('decay_rate', 1.0)

    for book_id, roundtrip_volume in miner_roundtrip_volumes.items():
        if latest_roundtrip_volumes[book_id] > 0:
            activity_factors_uid[book_id] = min(1 + ((roundtrip_volume * volume_cap_inv) * activity_impact), 2.0)
        else:
            if decay_rate == 0.0:
                continue
            latest_time = latest_roundtrip_timestamps[book_id]
            
            if latest_time > 0:
                inactive_time = max(0, simulation_timestamp - latest_time)
            else:
                inactive_time = simulation_timestamp
            
            current_factor = activity_factors_uid[book_id]
            activity_multiplier = max(current_factor, 1.0)

            if inactive_time <= decay_grace_period:
                time_acceleration = 1.0
            else:
                time_beyond_grace = inactive_time - decay_grace_period
                time_ratio = time_beyond_grace * decay_window_ns_inv
                time_acceleration = 1 + (time_ratio ** time_acceleration_power) * decay_rate
            
            total_acceleration = activity_multiplier * time_acceleration
            total_acceleration = min(total_acceleration, 100.0)
            
            try:
                decay_factor = base_decay_factor ** total_acceleration
                if not np.isfinite(decay_factor):
                    bt.logging.error(f"UID {uid} book {book_id}: Non-finite decay_factor")
                    decay_factor = 0.0
            except (OverflowError, ValueError) as e:
                bt.logging.error(f"UID {uid} book {book_id}: Decay overflow - {e}")
                decay_factor = 0.0
            
            activity_factors_uid[book_id] *= decay_factor
    
    # Calculate activity weighted normalized kappas - includes ALL books (None kappas = 0.0)
    activity_weighted_normalized_kappas = []
    for book_id, activity_factor in activity_factors_uid.items():
        norm_kappa = normalized_kappas[book_id]  # This is 0.0 if kappa was None
        if activity_factor < 1 or norm_kappa > 0.5:
            weighted = activity_factor * norm_kappa
        else:
            weighted = (2 - activity_factor) * norm_kappa
        activity_weighted_normalized_kappas.append(min(weighted, 1))

    uid_kappa['books_weighted'] = {
        book_id: weighted_kappa
        for book_id, weighted_kappa in enumerate(activity_weighted_normalized_kappas)
    }
    
    # Use the 1.5 rule to detect left-hand outliers in the activity-weighted Kappas    
    data = np.array(activity_weighted_normalized_kappas)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    # Apply minimum IQR and scale penalty to reward consistency
    min_iqr = 0.01

    effective_iqr = max(iqr, min_iqr)
    lower_threshold = q1 - 1.5 * effective_iqr
    outliers = data[data < lower_threshold]
    
    # Outliers detected here are activity-weighted Kappas which are significantly lower than those achieved on other books
    # A penalty equal to 67% of the difference between the mean outlier value and the value at the centre of the possible activity weighted Kappa values is calculated
    # Penalty is scaled by consistency: tight clusters (low IQR) get reduced penalty to reward consistent performance
    if len(outliers) > 0 and np.median(outliers) < 0.5:
        base_penalty = (0.5 - np.median(outliers)) / 1.5
        consistency_bonus = 1.0 - np.exp(-5 * iqr)  # Sigmoid-like scaling
        outlier_penalty = base_penalty * consistency_bonus
    else:
        outlier_penalty = 0
    
    # The median of the activity weighted Kappas provides the base score for the miner
    # This now includes books with None kappa (treated as 0.0), so inactive books pull down the median
    activity_weighted_normalized_median = np.median(data)
    # The penalty factor is subtracted from the base score to punish particularly poor performance on any particular book
    kappa_score = max(
        activity_weighted_normalized_median - abs(outlier_penalty), 
        0.0
    )

    uid_kappa['activity_weighted_normalized_median'] = activity_weighted_normalized_median
    uid_kappa['penalty'] = abs(outlier_penalty)
    uid_kappa['score'] = kappa_score
    
    return kappa_score

def score_uids(validator_data: Dict) -> Dict:
    """
    Calculates the new score value for all UIDs by computing realized Kappa-3 ratios only.

    This function orchestrates the Kappa-3 calculation process:
    1. Extracts realized P&L history (from completed trades)
    2. Calls kappa_3() or batch_kappa_3() to compute metrics
    3. Calls score_uid() to combine scores with activity weighting

    Args:
        validator_data (Dict): Dictionary containing validator state with keys:
            - kappa_values: Storage for Kappa-3 metrics
            - realized_pnl_history: Realized P&L from completed trades
            - config: Scoring configuration (includes min_realized_observations and tau)
            - uids: List of UIDs to process
            - deregistered_uids: UIDs pending reset
            - simulation_config: Simulation parameters

    Returns:
        Dict: Final scores for all UIDs
            Format: {uid: score}
    """
    config = validator_data['config']['scoring']
    uids = validator_data['uids']
    deregistered_uids = validator_data['deregistered_uids']
    simulation_config = validator_data['simulation_config']
    kappa_values = validator_data['kappa_values']
    kappa_cache = validator_data['kappa_cache']
    realized_pnl_history = validator_data['realized_pnl_history']
    tau = config['kappa']['tau']

    if config['kappa']['parallel_workers'] == 0:
        cache_updates = {}
        for uid in uids:
            realized_pnl_value = realized_pnl_history.get(uid, {})
            kappa_result = kappa_3(
                uid,
                realized_pnl_value,
                tau,
                config['kappa']['lookback'],
                config['kappa']['normalization_min'],
                config['kappa']['normalization_max'],
                config['kappa']['min_lookback'],
                config['kappa']['min_realized_observations'],
                simulation_config['grace_period'],
                deregistered_uids,
                simulation_config['book_count'],
                cache=kappa_cache
            )
            kappa_values[uid] = kappa_result
            fingerprint = _get_pnl_fingerprint(realized_pnl_value)
            cache_updates[uid] = (fingerprint, kappa_result)
        kappa_cache.update(cache_updates)        
    else:
        if config['kappa']['parallel_workers'] == -1:
            num_processes = len(config['kappa']['reward_cores'])
        else:
            num_processes = config['kappa']['parallel_workers']
        batch_size = int(256 / num_processes)
        batches = [uids[i:i+batch_size] for i in range(0, 256, batch_size)]
        kappa_results, cache_updates = batch_kappa_3(
            realized_pnl_history,
            tau,
            batches,
            config['kappa']['lookback'],
            config['kappa']['normalization_min'],
            config['kappa']['normalization_max'],
            config['kappa']['min_lookback'],
            config['kappa']['min_realized_observations'],
            simulation_config['grace_period'],
            deregistered_uids,
            simulation_config['book_count'],
            cache=kappa_cache,
            cores=config['kappa']['reward_cores'][:num_processes]            
        )
        kappa_values.update(kappa_results)
        kappa_cache.update(cache_updates)

    validator_data['kappa_values'] = kappa_values

    uid_scores = {
        uid: score_uid(validator_data, uid)
        for uid in uids
    }
    return uid_scores

def distribute_rewards(rewards: list, config: Dict) -> torch.FloatTensor:
    """
    Distributes rewards using a Pareto distribution to create variance in reward allocation.

    Args:
        rewards (list): List of raw reward scores for each UID
        config (Dict): Configuration dictionary containing rewarding parameters with keys:
            - rewarding.seed (int): Random seed for reproducibility
            - rewarding.pareto.shape (float): Shape parameter for Pareto distribution
            - rewarding.pareto.scale (float): Scale parameter for Pareto distribution

    Returns:
        torch.FloatTensor: Tensor of distributed rewards maintaining the original order of UIDs
    """
    rng = np.random.default_rng(config['rewarding']['seed'])
    num_uids = len(rewards)
    distribution = torch.FloatTensor(sorted(
        config['rewarding']['pareto']['scale'] *
        rng.pareto(config['rewarding']['pareto']['shape'], num_uids)
    ))
    rewards_tensor = torch.FloatTensor(rewards)
    sorted_rewards, sorted_indices = rewards_tensor.sort()
    distributed_rewards = distribution * sorted_rewards
    return torch.gather(distributed_rewards, 0, sorted_indices.argsort())

def get_rewards(self: 'Validator') -> Tuple[torch.FloatTensor, Dict]:
    """
    Calculate rewards.

    Args:
        validator (Validator): Validator instance

    Returns:
        Tuple[torch.FloatTensor, Dict]: (rewards, updated_data)
    """
    roundtrip_volumes = self.roundtrip_volumes
    realized_pnl_history = self.realized_pnl_history
    validator_data = {
        'kappa_values': self.kappa_values,
        'kappa_cache': self.kappa_cache,
        'activity_factors': self.activity_factors,
        'roundtrip_volumes': roundtrip_volumes,
        'realized_pnl_history': realized_pnl_history,
        'config': {
            'scoring': {
                'kappa': {
                    'normalization_min': self.config.scoring.kappa.normalization_min,
                    'normalization_max': self.config.scoring.kappa.normalization_max,
                    'min_lookback': self.config.scoring.kappa.min_lookback,
                    'lookback': self.config.scoring.kappa.lookback,
                    'min_realized_observations': self.config.scoring.kappa.min_realized_observations,
                    'parallel_workers': self.config.scoring.kappa.parallel_workers,
                    'reward_cores': self.reward_cores,
                    'tau': self.config.scoring.kappa.tau,
                },
                'activity': {
                    'capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                    'trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                    'trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
                    'decay_grace_period': self.config.scoring.activity.decay_grace_period,
                    'impact' : self.config.scoring.activity.impact,
                    'decay_rate': self.config.scoring.activity.decay_rate
                },
                'interval': self.config.scoring.interval,
            },
            'rewarding': {
                'seed': self.config.rewarding.seed,
                'pareto': {
                    'shape': self.config.rewarding.pareto.shape,
                    'scale': self.config.rewarding.pareto.scale,
                }
            },
        },
        'simulation_config': {
            'miner_wealth': self.simulation.miner_wealth,
            'publish_interval': self.simulation.publish_interval,
            'volumeDecimals': self.simulation.volumeDecimals,
            'grace_period': self.simulation.grace_period,
            'book_count': self.simulation.book_count, 
        },
        'simulation_timestamp': self.simulation_timestamp,
        'uids': [uid.item() for uid in self.metagraph.uids],
        'deregistered_uids': self.deregistered_uids,
        'device': self.device,
    }
    
    uid_scores = score_uids(validator_data)
    rewards = list(uid_scores.values())
    distributed_rewards = distribute_rewards(rewards, validator_data['config']).to(self.device)
    
    updated_data = {
        'kappa_values': validator_data['kappa_values'],
        'activity_factors': validator_data['activity_factors']
    }
    
    return distributed_rewards, updated_data

def set_delays(self: 'Validator', synapse_responses: dict[int, MarketSimulationStateUpdate]) -> list[FinanceAgentResponse]:
    """
    Applies base delay based on process time using an exponential mapping,
    and adds a per-book Gaussian-distributed random latency instruction_delay to instructions,
    with zero instruction_delay applied to the first instruction per book.

    Args:
        self (taos.im.neurons.validator.Validator): Validator instance.
        synapse_responses (dict[int, MarketSimulationStateUpdate]): Latest state updates.

    Returns:
        list[FinanceAgentResponse]: Delayed finance responses.
    """
    start_time = time.time()
    responses = []
    timeout = self.config.neuron.timeout
    min_delay = self.config.scoring.min_delay
    max_delay = self.config.scoring.max_delay
    min_instruction_delay = self.config.scoring.min_instruction_delay
    max_instruction_delay = self.config.scoring.max_instruction_delay

    def compute_delay(p_time: float) -> int:
        """Exponential scaling of process time into delay."""
        t = p_time / timeout
        exp_scale = 5
        delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
        delay = min_delay + delay_frac * (max_delay - min_delay)
        return int(delay)
    log_messages = []
    for uid, synapse_response in synapse_responses.items():
        response = synapse_response.response
        if response:
            base_delay = compute_delay(synapse_response.dendrite.process_time)
            seen_books = set()
            for instruction in response.instructions:
                book_id = instruction.bookId

                if book_id not in seen_books:
                    instruction_delay = 0
                    seen_books.add(book_id)
                else:
                    instruction_delay = random.randint(min_instruction_delay, max_instruction_delay)
                instruction.delay += base_delay + instruction_delay
            responses.append(response)
            log_messages.append(
                f"UID {response.agent_id} responded with {len(response.instructions)} instructions "
                f"after {synapse_response.dendrite.process_time:.4f}s – base delay {base_delay}{self.simulation.time_unit}"
            )
    if log_messages:
        bt.logging.info("\n".join(log_messages))    
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        bt.logging.warning(f"set_delays took {elapsed:.4f}s for {len(synapse_responses)} responses")
    return responses