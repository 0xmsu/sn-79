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

def calculate_kappa_score(
    uid: int,
    kappa_values: Dict,
    activity_factors: Dict,
    pnl_factors: Dict,
    roundtrip_volumes: Dict,
    realized_pnl_history: Dict,
    config: Dict,
    simulation_config: Dict,
    simulation_timestamp: int
) -> float:
    """
    Calculate Kappa-based score with activity and P&L factor weighting.
    
    This function measures risk-adjusted trading quality (Kappa-3 metric) and weights it by:
    1. Trading activity (volume-based): Rewards active participation
    2. Profitability (P&L-based): Boosts scores for profitable trading
    
    The Kappa score component represents "quality of trading WHERE the miner traded."
    Activity and P&L factors are applied per-book to weight the quality measure.
    
    Scoring Flow:
    1. Normalize raw Kappa values per book to [0, 1] range
    2. Calculate volume-based activity factors per book (decay for inactivity)
    3. Calculate P&L-based multipliers per book (if enabled via config)
    4. Combine activity × P&L factors and apply to normalized Kappa per book
    5. Aggregate weighted Kappas across books with outlier penalty
    6. Return final Kappa score in [0, 1] range
    
    Args:
        uid: UID of miner being scored
        kappa_values: Kappa calculation results from kappa_3() {uid: {books: {book_id: kappa_value}}}
        activity_factors: Activity factor storage (updated in-place) {uid: {book_id: factor}}
        pnl_factors: P&L factor storage (updated in-place) {uid: {book_id: factor}}
        roundtrip_volumes: Trading volume history {uid: {book_id: {timestamp: volume}}}
        realized_pnl_history: Realized P&L from completed trades {uid: {timestamp: {book_id: pnl}}}
        config: Scoring configuration dict
        simulation_config: Simulation parameters (miner_wealth, publish_interval, etc)
        simulation_timestamp: Current simulation timestamp in nanoseconds
        
    Returns:
        float: Kappa score in [0, 1] range
    """
    # Early exit if no Kappa data for this miner
    if not kappa_values[uid]:
        return 0.0

    uid_kappa = kappa_values[uid]
    kappas = uid_kappa['books']  # Raw Kappa values per book from kappa_3()
    
    # ===== STEP 1: NORMALIZE KAPPA VALUES =====
    # Maps raw Kappa values to [0, 1] range for fair comparison
    # Normalization allows combining Kappas from different books on a common scale
    norm_min = config['kappa']['normalization_min']
    norm_max = config['kappa']['normalization_max']
    norm_range = norm_max - norm_min
    norm_range_inv = 1.0 / norm_range if norm_range != 0 else 0.0
    
    # Normalize kappas per book - keep None for books with insufficient data
    # None indicates the miner hasn't traded enough on that book to calculate reliable Kappa
    normalized_kappas = {book_id: None for book_id in activity_factors[uid].keys()}
    
    for book_id, kappa_val in kappas.items():
        if kappa_val is not None:
            # Clamp to [0, 1] range: (kappa - min) / (max - min)
            normalized_kappas[book_id] = max(0.0, min(1.0, (kappa_val - norm_min) * norm_range_inv))

    # ===== STEP 2: CALCULATE ACTIVITY FACTOR PARAMETERS =====
    # Activity factors reward miners for trading volume relative to their capital
    # Volume cap together with impact parameter defines the threshold for maximum activity boost
    volume_cap = round(
        config['activity']['capital_turnover_cap'] * simulation_config['miner_wealth'],
        simulation_config['volumeDecimals']
    )
    volume_cap_inv = 1.0 / volume_cap

    lookback = config['kappa']['lookback']  # Number of intervals to look back
    publish_interval = simulation_config['publish_interval']  # Nanoseconds per interval
    lookback_threshold = simulation_timestamp - (lookback * publish_interval)
    
    # Decay parameters: Activity factors decay for inactive books to incentivize consistent trading
    decay_grace_period = config['activity'].get('decay_grace_period', 600_000_000_000)
    activity_impact = config['activity'].get('impact', 0.33)  # Max boost from activity
    time_acceleration_power = 2.0  # Exponential decay acceleration
    
    # Calculate decay parameters
    # The decay window is the lookback period minus the grace period
    # During grace period: no decay (allows brief pauses without penalty)
    # After grace period: exponential decay kicks in
    scoring_interval_seconds = config['interval'] / 1e9
    total_intervals = lookback // scoring_interval_seconds
    grace_intervals = decay_grace_period / config['interval']
    decay_window_intervals = total_intervals - grace_intervals
    
    # Safety check for decay window
    if decay_window_intervals <= 0:
        bt.logging.warning(f"UID {uid}: Invalid decay window (total={total_intervals}, grace={grace_intervals})")
        base_decay_factor = 0.999  # Fallback to very slow decay
    else:
        # Calculate base decay factor: 2^(-1/decay_window_intervals)
        # This creates exponential decay that reaches ~0.5 after decay_window_intervals periods
        base_decay_factor = 2 ** (-1 / decay_window_intervals)
        base_decay_factor = max(0.5, min(0.9999, base_decay_factor))  # Clamp to safe range
    
    decay_window_ns = (lookback * config['interval']) - decay_grace_period
    decay_window_ns_inv = 1.0 / decay_window_ns if decay_window_ns > 0 else 0.0

    # ===== STEP 3: COMPUTE ROUNDTRIP VOLUMES PER BOOK =====
    # Roundtrip volume = sum of trade sizes for completed buy-sell cycles
    # This measures actual trading activity, not just open positions
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
                    
                    # Sum all volumes within lookback period
                    # Find the most recent trading timestamp
                    for ts, vol in rt_volumes.items():
                        if ts >= lookback_threshold:
                            lookback_volume += vol
                        if vol > 0 and ts <= sampled_timestamp and ts > latest_time:
                            latest_time = ts
                    
                    # Check if there was recent activity (within sampling interval)
                    if latest_time > 0 and latest_time >= sampled_timestamp - sampling_interval:
                        latest_volume = rt_volumes[latest_time]
                    
                    miner_roundtrip_volumes[book_id] = lookback_volume
                    latest_roundtrip_volumes[book_id] = latest_volume
                    latest_roundtrip_timestamps[book_id] = latest_time
                else:
                    # No volume history for this book
                    miner_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_timestamps[book_id] = 0
            else:
                # Book not in roundtrip volumes
                miner_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_timestamps[book_id] = 0
    else:
        # UID not in roundtrip_volumes, initialize all books to zero
        for book_id in range(num_books):
            miner_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_timestamps[book_id] = 0

    # ===== STEP 4: CALCULATE VOLUME-BASED ACTIVITY FACTORS =====
    # Activity factors range from 0 to 2.0:
    # - 1.0 = neutral (no boost or penalty)
    # - <1.0 = penalty for inactivity (via exponential decay)
    # - >1.0 = boost for high volume (up to 2.0x at volume_cap)
    activity_factors_uid = activity_factors[uid]
    decay_rate = config['activity'].get('decay_rate', 1.0)

    for book_id, roundtrip_volume in miner_roundtrip_volumes.items():
        if latest_roundtrip_volumes[book_id] > 0:
            # ACTIVE BOOK: Calculate activity boost based on volume
            # Formula: 1 + (volume/volume_cap × activity_impact)
            activity_factors_uid[book_id] = min(
                1 + ((roundtrip_volume * volume_cap_inv) * activity_impact), 
                2.0
            )
        else:
            # INACTIVE BOOK: Apply exponential decay
            if decay_rate == 0.0:
                # Decay disabled, skip this book
                continue
                
            latest_time = latest_roundtrip_timestamps[book_id]
            
            if latest_time > 0:
                # Calculate time since last activity
                inactive_time = max(0, simulation_timestamp - latest_time)
            else:
                # Never traded on this book, use full simulation time
                inactive_time = simulation_timestamp
            
            current_factor = activity_factors_uid[book_id]
            activity_multiplier = max(current_factor, 1.0)

            # Time acceleration: Decay accelerates based on how long inactive
            if inactive_time <= decay_grace_period:
                # Within grace period: no decay acceleration
                time_acceleration = 1.0
            else:
                # Beyond grace period: accelerated decay
                # The longer inactive, the faster the decay
                time_beyond_grace = inactive_time - decay_grace_period
                time_ratio = time_beyond_grace * decay_window_ns_inv
                # Quadratic acceleration (time_acceleration_power = 2.0)
                time_acceleration = 1 + (time_ratio ** time_acceleration_power) * decay_rate
            
            # Total acceleration combines current factor with time acceleration
            # Higher current factors decay faster (to prevent "coasting" on past activity)
            total_acceleration = activity_multiplier * time_acceleration
            total_acceleration = min(total_acceleration, 100.0)  # Safety cap
            
            # Apply exponential decay: factor *= base_decay_factor^acceleration
            try:
                decay_factor = base_decay_factor ** total_acceleration
                if not np.isfinite(decay_factor):
                    bt.logging.error(f"UID {uid} book {book_id}: Non-finite decay_factor")
                    decay_factor = 0.0
            except (OverflowError, ValueError) as e:
                bt.logging.error(f"UID {uid} book {book_id}: Decay overflow - {e}")
                decay_factor = 0.0
            
            # Update activity factor with decay
            activity_factors_uid[book_id] *= decay_factor
    
    # ===== STEP 5: CALCULATE P&L-BASED MULTIPLIERS =====
    # P&L factors boost/penalize Kappa scores based on realized profitability per book
    # This rewards miners who achieve good risk-adjusted returns AND make money
    # Range: 0.0 to (1 + config.kappa.pnl.impact)
    pnl_factors_uid = pnl_factors[uid]
    pnl_impact = config.get('kappa', {}).get('pnl', {}).get('impact', 0.0)
    
    if pnl_impact > 0:
        # Normalization: 100% DAILY return is the baseline for max boost
        # This makes P&L factors comparable across different assessment windows
        DAILY_NS = 86400_000_000_000  # 24 hours in nanoseconds
        assessment_window_ns = lookback * config['interval']
        window_fraction = assessment_window_ns / DAILY_NS
        pnl_reference = simulation_config['miner_wealth'] * window_fraction
        
        # Compute realized P&L per book over lookback window
        # Only counts completed trades (realized gains/losses)
        book_realized_pnl = {}
        if uid in realized_pnl_history:
            for timestamp, books_dict in realized_pnl_history[uid].items():
                if timestamp >= lookback_threshold:
                    for book_id, pnl in books_dict.items():
                        if book_id not in book_realized_pnl:
                            book_realized_pnl[book_id] = 0.0
                        book_realized_pnl[book_id] += pnl
        
        # Calculate P&L factor per book
        for book_id in normalized_kappas.keys():
            realized_pnl_book = book_realized_pnl.get(book_id, 0.0)
            
            # Normalize by daily-scaled reference
            # pnl_ratio = 1.0 means they earned enough to imply 100% daily return
            pnl_ratio = realized_pnl_book / pnl_reference
            
            # Raw P&L factor calculation:
            #   0% daily return → 1.0x (neutral)
            #   +100% daily return → (1+impact)x
            #   -100% daily return → (1+impact)x
            raw_pnl_factor = max(1.0 + pnl_ratio, 0.0)
            
            # Apply impact scaling: controls how much P&L affects final score
            pnl_factor = 1.0 + ((raw_pnl_factor - 1.0) * pnl_impact)
            
            # Cap at 2x boost
            pnl_factor = min(pnl_factor, 2.0)
            
            pnl_factors_uid[book_id] = pnl_factor
    else:
        # P&L weighting disabled - set all to neutral (1.0 = no effect)
        for book_id in normalized_kappas.keys():
            pnl_factors_uid[book_id] = 1.0
    
    # ===== STEP 6: COMBINE ACTIVITY AND P&L FACTORS, APPLY TO KAPPA =====
    # Both factors are multiplicative: combined_factor = activity × pnl
    # This means miners need BOTH good activity AND good profitability for max boost
    # Asymmetric weighting is applied based on factor magnitude and Kappa value
    activity_weighted_normalized_kappas = {}
    
    for book_id in normalized_kappas.keys():
        activity_factor = activity_factors_uid[book_id]
        pnl_factor = pnl_factors_uid[book_id]
        combined_factor = activity_factor * pnl_factor
        
        norm_kappa = normalized_kappas[book_id]
        
        if norm_kappa is None:
            # No Kappa data for this book (insufficient trading history)
            activity_weighted_normalized_kappas[book_id] = None
        else:
            # Apply combined factor with asymmetric weighting
            if combined_factor < 1 or norm_kappa > 0.5:
                weighted = combined_factor * norm_kappa
            else:
                weighted = (2 - combined_factor) * norm_kappa
            
            # Cap at 1.0 (normalized Kappa range)
            activity_weighted_normalized_kappas[book_id] = min(weighted, 1)

    uid_kappa['books_weighted'] = activity_weighted_normalized_kappas
    
    # ===== STEP 7: HANDLE INACTIVE BOOKS =====
    # Miners are allowed a certain number of inactive books without penalty
    max_inactive_books = config.get('max_inactive_books', 24)
    
    # Separate books into: valid kappa vs no kappa (None)
    books_with_scores = []
    books_with_no_kappa = []
    
    for book_id, score in activity_weighted_normalized_kappas.items():
        if score is None:
            books_with_no_kappa.append(book_id)
        else:
            books_with_scores.append(score)
    
    num_books_no_kappa = len(books_with_no_kappa)
    
    # Determine scoring data
    if num_books_no_kappa <= max_inactive_books:
        # Can ignore all books with no kappa (within allowed limit)
        data = np.array(books_with_scores)
        
        uid_kappa['inactive_books'] = books_with_no_kappa
        uid_kappa['penalty_books'] = []
        
        bt.logging.trace(
            f"UID {uid}: Ignoring {num_books_no_kappa} books with no kappa (≤ max_inactive_books={max_inactive_books})"
        )
    else:
        # More than max_inactive_books have no kappa - excess contribute 0.0
        excess_inactive = num_books_no_kappa - max_inactive_books
        
        # Include valid scores + zeros for excess inactive books
        data = np.array(books_with_scores + [0.0] * excess_inactive)
        
        uid_kappa['inactive_books'] = books_with_no_kappa[:max_inactive_books]
        uid_kappa['penalty_books'] = books_with_no_kappa[max_inactive_books:]
        
        bt.logging.debug(
            f"UID {uid}: {num_books_no_kappa} books with no kappa (> max_inactive_books={max_inactive_books}). "
            f"Ignoring {max_inactive_books}, penalizing {excess_inactive} as 0.0"
        )
    
    # ===== STEP 8: CALCULATE OUTLIER PENALTY =====
    # Use the 1.5×IQR rule to detect left-hand outliers in the activity-weighted Kappas
    # Outliers indicate books where the miner performed significantly worse than their median
    # This penalizes inconsistent performance across books
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    # Apply minimum IQR to prevent division issues and scale penalty appropriately
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
    
    # ===== STEP 9: FINAL KAPPA SCORE =====
    # The median of the activity-weighted Kappas provides the base score for the miner
    # Median is robust to outliers and represents "typical" performance across books
    activity_weighted_normalized_median = np.median(data)
    
    # The penalty factor is subtracted from the base score to punish particularly  poor performance on any particular book
    kappa_score = max(activity_weighted_normalized_median - abs(outlier_penalty), 0.0)

    # Store metadata for debugging/reporting
    uid_kappa['activity_weighted_normalized_median'] = activity_weighted_normalized_median
    uid_kappa['penalty'] = abs(outlier_penalty)
    uid_kappa['score'] = kappa_score
    uid_kappa['num_scored_books'] = len(data)
    
    return kappa_score


def calculate_pnl_score(
    realized_pnl_history: Dict,
    uid: int,
    lookback: int,
    interval: int,
    lookback_threshold: int,
    miner_wealth: float,
    config: Dict
) -> float:
    """
    Calculate normalized P&L score using daily return normalization.
    
    This function measures absolute profitability independent of risk considerations.
    
    The score represents aggregate profitability:
    - Sums ALL realized P&L across ALL books over the lookback window
    - Normalizes to equivalent daily return rate (makes comparable across time windows)
    - Maps to [0, 1] with 0.5 as neutral (breakeven)
    
    Scoring Philosophy:
    - Kappa component: "Quality WHERE you traded" (per-book, activity-weighted)
    - P&L component: "Total profit made" (aggregate, no activity weighting)
    - Combination: Balances risk-adjusted quality with absolute returns
    
    Args:
        realized_pnl_history: P&L history from completed trades
            Format: {uid: {timestamp: {book_id: pnl}}}
        uid: UID to score
        lookback: Lookback period in intervals
        interval: Interval duration in nanoseconds
        lookback_threshold: Minimum timestamp to include (pre-calculated)
        miner_wealth: Initial capital for normalization
        config: Normalization config with optional keys:
            - min_daily_return: Floor for daily return (default: -1.0 = -100%)
            - max_daily_return: Cap for daily return (default: 1.0 = +100%)
        
    Returns:
        float: P&L score in [0, 1] range
    """
    # Early exit if no P&L data
    if uid not in realized_pnl_history:
        return 0.5  # Neutral score for no P&L data
    
    # ===== STEP 1: SUM TOTAL REALIZED P&L =====
    # Aggregate ALL realized P&L across ALL books within lookback window
    total_realized_pnl = 0.0
    for timestamp, books_dict in realized_pnl_history[uid].items():
        if timestamp >= lookback_threshold:
            for book_pnl in books_dict.values():
                total_realized_pnl += book_pnl
    
    # ===== STEP 2: NORMALIZE TO DAILY RETURN RATE =====
    # Convert absolute P&L to daily return percentage for fair comparison
    # This makes P&L scores comparable across different assessment windows
    #
    # Formula: daily_return_ratio = (total_pnl) / (capital × days)
    # Where days = assessment_window / 24_hours
    DAILY_NS = 86400_000_000_000  # 24 hours in nanoseconds
    assessment_window_ns = lookback * interval
    window_fraction = assessment_window_ns / DAILY_NS  # Fraction of a day
    pnl_reference = miner_wealth * window_fraction
    
    if pnl_reference == 0:
        bt.logging.warning(f"UID {uid}: P&L reference is zero (lookback={lookback}, interval={interval})")
        return 0.5
    
    # Calculate daily return ratio
    daily_return_ratio = total_realized_pnl / pnl_reference
    
    # ===== STEP 3: CLIP TO EXPECTED BOUNDS =====
    # Get normalization bounds from config (defaults: ±100% daily)
    # Clipping prevents extreme outliers from dominating the score distribution
    min_daily = config.get('min_daily_return', -1.0)
    max_daily = config.get('max_daily_return', 1.0)
    daily_return_ratio = max(min_daily, min(max_daily, daily_return_ratio))
    
    # ===== STEP 4: MAP TO [0, 1] SCORE =====
    # Map daily return to score with 0% return at 0.5 (neutral)
    # Formula: score = 0.5 + (daily_return / 2.0)
    pnl_score = 0.5 + (daily_return_ratio / 2.0)
    
    # Ensure strict [0, 1] bounds (safety check)
    pnl_score = max(0.0, min(1.0, pnl_score))
    
    bt.logging.trace(
        f"UID {uid}: P&L score calculation - "
        f"total_pnl={total_realized_pnl:.2f}, "
        f"daily_return={daily_return_ratio*100:.1f}%, "
        f"score={pnl_score:.4f}"
    )
    
    return pnl_score


def score_uid(validator_data: Dict, uid: int) -> float:
    """
    Calculates the final combined score for a specific UID.
    
    This function orchestrates the complete scoring pipeline by combining:
    1. Kappa Score: Risk-adjusted returns weighted by activity and P&L factors (per-book)
    2. P&L Score: Absolute profitability normalized to daily return (aggregate)
    
    Scoring Philosophy:
    -------------------
    The final score balances two complementary dimensions:
    
    KAPPA COMPONENT (measures "HOW WELL" you trade):
    - Per-book risk-adjusted returns (Kappa-3 metric)
    - Weighted by per-book activity factors (rewards consistent participation)
    - Weighted by per-book P&L factors (boosts for profitability)
    - Aggregated across books with outlier penalty
    
    P&L COMPONENT (measures "HOW MUCH" you make):
    - Total realized profit/loss across all books
    - Normalized to daily return rate
    
    COMBINATION:
    Final Score = (kappa_weight × kappa_score) + (pnl_weight × pnl_score)

    Args:
        validator_data (Dict): Dictionary containing validator state with keys:
            - kappa_values: Kappa calculation results
            - activity_factors: Per-book activity factor storage
            - pnl_factors: Per-book P&L factor storage
            - roundtrip_volumes: Trading volume history
            - realized_pnl_history: Realized P&L from completed trades
            - config: Scoring configuration
            - simulation_config: Simulation parameters
            - simulation_timestamp: Current timestamp
        uid (int): UID of miner being scored

    Returns:
        float: The final combined score for the given UID in [0, 1] range
        
    Side Effects:
        - Updates uid_kappa dict with scoring metadata:
            - kappa_score: Base Kappa score (if P&L enabled)
            - pnl_score: P&L score (if enabled, else None)
            - kappa_weight: Weight applied to Kappa component
            - pnl_score_weight: Weight applied to P&L component
            - final_score: Combined final score
    """
    # Extract required data from validator state
    kappa_values = validator_data['kappa_values']
    activity_factors = validator_data['activity_factors']
    pnl_factors = validator_data['pnl_factors']
    roundtrip_volumes = validator_data['roundtrip_volumes']
    realized_pnl_history = validator_data['realized_pnl_history']
    config = validator_data['config']['scoring']
    simulation_config = validator_data['simulation_config']
    simulation_timestamp = validator_data['simulation_timestamp']

    # ===== STEP 1: CALCULATE KAPPA SCORE COMPONENT =====
    # This calculates risk-adjusted returns with activity and P&L weighting
    # The Kappa score already includes:
    # - Per-book activity factors (volume-based participation weighting)
    # - Per-book P&L factors (profitability-based quality weighting)
    # - Outlier penalty (consistency enforcement)
    kappa_score = calculate_kappa_score(
        uid=uid,
        kappa_values=kappa_values,
        activity_factors=activity_factors,
        pnl_factors=pnl_factors,
        roundtrip_volumes=roundtrip_volumes,
        realized_pnl_history=realized_pnl_history,
        config=config,
        simulation_config=simulation_config,
        simulation_timestamp=simulation_timestamp
    )
    
    # ===== STEP 2: CHECK IF P&L SCORE COMPONENT IS ENABLED =====
    pnl_config = config.get('pnl', {})
    pnl_score_weight = pnl_config.get('score_weight', 0.0)
    
    if pnl_score_weight > 0:
        # P&L SCORE ENABLED: Calculate and combine both components        
        # Calculate lookback threshold (same as used for Kappa)
        lookback = config['kappa']['lookback']
        publish_interval = simulation_config['publish_interval']
        lookback_threshold = simulation_timestamp - (lookback * publish_interval)

        pnl_score = calculate_pnl_score(
            realized_pnl_history=realized_pnl_history,
            uid=uid,
            lookback=lookback,
            interval=config['interval'],
            lookback_threshold=lookback_threshold,
            miner_wealth=simulation_config['miner_wealth'],
            config=pnl_config.get('normalization', {})
        )
        
        # ===== STEP 3: COMBINE COMPONENTS WITH WEIGHTED AVERAGE =====
        # Kappa weight is the complement of P&L weight
        # This ensures weights sum to 1.0
        kappa_weight = config['kappa'].get('score_weight', 0.0)
        
        # Weighted combination: final = (w_k × kappa) + (w_p × pnl)
        final_score = (kappa_weight * kappa_score) + (pnl_score_weight * pnl_score)
        
        # Store metadata for debugging/reporting
        if kappa_values[uid]:
            uid_kappa = kappa_values[uid]
            uid_kappa['pnl_score'] = pnl_score
            uid_kappa['kappa_weight'] = kappa_weight
            uid_kappa['pnl_score_weight'] = pnl_score_weight
            uid_kappa['final_score'] = final_score
        
        bt.logging.trace(
            f"UID {uid}: Combined score - "
            f"kappa={kappa_score:.4f} (w={kappa_weight:.2f}, has activity), "
            f"pnl={pnl_score:.4f} (w={pnl_score_weight:.2f}, no activity), "
            f"final={final_score:.4f}"
        )
        
        return final_score
    else:
        # P&L SCORE DISABLED: Return Kappa score only
        if kappa_values[uid]:
            uid_kappa = kappa_values[uid]
            uid_kappa['pnl_score'] = None
            uid_kappa['kappa_weight'] = 1.0
            uid_kappa['pnl_score_weight'] = 0.0
            uid_kappa['final_score'] = kappa_score
        
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
        total_uids = len(uids)
        batch_size = max(1, total_uids // num_processes)
        batches = []
        for i in range(0, total_uids, batch_size):
            batch = uids[i:i+batch_size]
            if batch:
                batches.append(batch)
        actual_cores = config['kappa']['reward_cores'][:len(batches)]
        bt.logging.debug(
            f"Parallel kappa calculation: {total_uids} UIDs split into {len(batches)} batches "
            f"(batch sizes: {[len(b) for b in batches]}) across {len(actual_cores)} cores"
        )
        
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
            cores=actual_cores
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
    all_uids = list(range(self.effective_max_uids))

    validator_data = {
        'kappa_values': self.kappa_values,
        'kappa_cache': self.kappa_cache,
        'activity_factors': self.activity_factors,
        'pnl_factors': self.pnl_factors,
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
                    'pnl_impact' : self.config.scoring.kappa.pnl.impact
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
        'uids': all_uids,
        'deregistered_uids': self.deregistered_uids,
        'device': self.device,
    }
    
    uid_scores = score_uids(validator_data)
    rewards = list(uid_scores.values())
    distributed_rewards = distribute_rewards(rewards, validator_data['config']).to(self.device)
    
    updated_data = {
        'kappa_values': validator_data['kappa_values'],
        'activity_factors': validator_data['activity_factors'],
        'pnl_factors': validator_data['pnl_factors']
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
                f"after {synapse_response.dendrite.process_time:.4f}s – base delay {base_delay}{self.simulation.time_unit if hasattr(self, 'simulation') else 'ns'}"
            )
    if log_messages:
        bt.logging.info("\n".join(log_messages))    
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        bt.logging.warning(f"set_delays took {elapsed:.4f}s for {len(synapse_responses)} responses")
    return responses