# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import traceback
import time
import torch
import psutil
import asyncio
import bittensor as bt
import pandas as pd

from typing import Dict
from taos.im.neurons.validator import Validator
from taos.im.protocol.models import TradeInfo

from taos.common.utils.prometheus import prometheus
from taos.im.utils import duration_from_timestamp
from prometheus_client import Counter, Gauge, Info

def init_metrics(self : Validator) -> None:
    """
    Set up prometheus metric objects.

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
    Returns:
        None
    """
    prometheus (
        config = self.config,
        port = self.config.prometheus.port,
        level = None
    )
    self.prometheus_counters = Counter('counters', 'Counter summaries for the running validator.', ['wallet', 'netuid', 'timestamp', 'counter_name'])
    self.prometheus_simulation_gauges = Gauge('simulation_gauges', 'Gauge summaries for global simulation metrics.', ['wallet', 'netuid', 'simulation_gauge_name'])
    self.prometheus_validator_gauges = Gauge('validator_gauges', 'Gauge summaries for validator-related metrics.', ['wallet', 'netuid', 'validator_gauge_name'])
    self.prometheus_miner_gauges = Gauge('miner_gauges', 'Gauge summaries for miner-related metrics.', ['wallet', 'netuid', 'agent_id', 'miner_gauge_name'])
    self.prometheus_book_gauges = Gauge('book_gauges', 'Gauge summaries for book-related metrics.', ['wallet', 'netuid', 'book_id', 'level', 'book_gauge_name'])
    self.prometheus_agent_gauges = Gauge('agent_gauges', 'Gauge summaries for agent-related metrics.', ['wallet', 'netuid', 'book_id', 'agent_id', 'agent_gauge_name'])

    self.prometheus_trades = Gauge('trades', 'Gauge summaries for trade metrics.', [
        'wallet', 'netuid', 'timestamp', 'timestamp_str', 'book_id', 'agent_id', 'trade_id',
        'aggressing_order_id', 'aggressing_agent_id', 'resting_order_id', 'resting_agent_id',
        'maker_fee', 'taker_fee',
        'price', 'volume', 'side', 'trade_gauge_name'])
    self.prometheus_miner_trades = Gauge('miner_trades', 'Gauge summaries for agent trade metrics.', [
        'wallet', 'netuid', 'timestamp', 'timestamp_str', 'book_id', 'uid',
        'role', 'price', 'volume', 'side', 'fee',
        'miner_trade_gauge_name'])
    self.prometheus_books = Gauge('books', 'Gauge summaries for book snapshot metrics.', [
        'wallet', 'netuid', 'timestamp', 'timestamp_str', 'book_id',
        'bid_5', 'bid_vol_5', 'bid_4', 'bid_vol_4', 'bid_3', 'bid_vol_3', 'bid_2', 'bid_vol_2', 'bid_1', 'bid_vol_1',
        'ask_5', 'ask_vol_5', 'ask_4', 'ask_vol_4', 'ask_3', 'ask_vol_3', 'ask_2', 'ask_vol_2', 'ask_1', 'ask_vol_1',
        'book_gauge_name'
    ])
    self.prometheus_miners = Gauge('miners', 'Gauge summaries for miner metrics.', [
        'wallet', 'netuid', 'timestamp', 'timestamp_str', 'agent_id',
        'placement', 'base_balance', 'base_loan', 'base_collateral', 'quote_balance', 'quote_loan', 'quote_collateral',
        'inventory_value', 'inventory_value_change', 'pnl', 'pnl_change',
        'min_daily_volume','activity_factor', 'sharpe', 'sharpe_penalty', 'sharpe_score', 'unnormalized_score', 'score',
        'miner_gauge_name'
    ])
    self.prometheus_info = Info('neuron_info', "Info summaries for the running validator.", ['wallet', 'netuid'])

def publish_validator_gauges(self : Validator):
    """
    Publishes validator-specific metrics to Prometheus gauges.
    
    Metrics include validator metagraph information (UID, stake, trust, dividends, emission, 
    last update, active status) and system resource usage (CPU, RAM, disk).
    
    Args:
        self (Validator): The intelligent markets simulation validator instance
        
    Returns:
        None
    """
    bt.logging.debug(f"Publishing validator metrics...")
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="uid").set( self.uid )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="stake").set( self.metagraph.stake[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="validator_trust").set( self.metagraph.validator_trust[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="dividends").set( self.metagraph.dividends[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="emission").set( self.metagraph.emission[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="last_update").set( self.current_block - self.metagraph.last_update[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="active").set( self.metagraph.active[self.uid] )
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    disk_info = psutil.disk_usage('/')
    disk_usage = disk_info.percent
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="cpu_usage_percent").set( cpu_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="ram_usage_percent").set( memory_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, validator_gauge_name="disk_usage_percent").set( disk_usage )

def publish_info(self : Validator) -> None:
    """
    Publishes static simulation and validator information metrics

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
    Returns:
        None
    """
    prometheus_info = {
        'uid': str(self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )) if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys else -1,
        'network': self.config.subtensor.network,
        'coldkey': str(self.wallet.coldkeypub.ss58_address),
        'coldkey_name': self.config.wallet.name,
        'hotkey': str(self.wallet.hotkey.ss58_address),
        'name': self.config.wallet.hotkey
    } | {
         f"simulation_{name}" : str(value) for name, value in self.simulation.model_dump().items() if name != 'logDir' and name != 'fee_policy'
    } | self.simulation.fee_policy.to_prom_info()
    self.prometheus_info.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid ).info (prometheus_info)
    publish_validator_gauges(self)

def _set_if_changed(gauge, value, *labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        *labels: Variable number of positional label values for the gauge
        
    Returns:
        None
    """
    try:
        current = gauge.labels(*labels)._value.get()
        if current != value:
            gauge.labels(*labels).set(value)
    except KeyError:
        gauge.labels(*labels).set(value)

def _set_if_changed_metric(gauge, value, **labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value using keyword labels.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        **labels: Variable number of keyword label-value pairs for the gauge
        
    Returns:
        None
    """
    try:
        current = gauge.labels(**labels)._value.get()
    except KeyError:
        current = None
    if current != value:
        gauge.labels(**labels).set(value)

def report_worker(validator_data: Dict, state_data: Dict) -> Dict:
    """
    Worker function for calculating metrics - picklable for ProcessPoolExecutor.

    Args:
        validator_data (Dict): Dictionary containing validator state
        state_data (Dict): Dictionary containing simulation state

    Returns:
        Dict: Calculated metrics to be applied back to validator
    """

    result = {
        'metrics': {},
        'updated_stats': {},
        'error': None
    }

    try:
        simulation_timestamp = validator_data['simulation_timestamp']
        step = validator_data['step']
        accounts = state_data['accounts']
        books = state_data['books']
        if not accounts:
            return result

        # Calculate daily volumes
        trade_volumes = validator_data['trade_volumes']
        volume_decimals = validator_data['simulation_config']['volumeDecimals']

        daily_volumes = {
            agentId: {
                bookId: {
                    role: round(
                        sum([volume for volume in trade_volumes[agentId][bookId][role].values()]),
                        volume_decimals
                    )
                    for role in ['total', 'maker', 'taker', 'self']
                }
                for bookId in range(validator_data['book_count'])
            }
            for agentId in accounts.keys()
        }

        # Calculate inventory history totals and PNL
        inventory_history = validator_data['inventory_history']
        total_inventory_history = {}
        pnl = {}

        for agentId in accounts.keys():
            if agentId < 0 or len(inventory_history[agentId]) < 3:
                continue
            total_inventory_history[agentId] = [
                sum(list(inventory_value.values()))
                for inventory_value in list(inventory_history[agentId].values())
            ]
            pnl[agentId] = total_inventory_history[agentId][-1] - total_inventory_history[agentId][0]

        # Calculate scores and placements
        scores = torch.FloatTensor(list(validator_data['scores'].values()))
        indices = scores.argsort(dim=-1, descending=True)
        placements = torch.empty_like(indices).scatter_(
            -1, indices, torch.arange(scores.size(-1), device=scores.device)
        )

        # Prepare metrics for each miner
        miner_metrics = {}
        for agentId, accounts_data in accounts.items():
            if agentId < 0 or len(inventory_history[agentId]) < 3:
                continue

            base_decimals = validator_data['simulation_config']['baseDecimals']
            quote_decimals = validator_data['simulation_config']['quoteDecimals']

            total_base_balance = round(
                sum([accounts_data[bookId]['bb']['t'] for bookId in books]),
                base_decimals
            )
            total_base_loan = round(
                sum([accounts_data[bookId]['bl'] for bookId in books]),
                base_decimals
            )
            total_base_collateral = round(
                sum([accounts_data[bookId]['bc'] for bookId in books]),
                base_decimals
            )
            total_quote_balance = round(
                sum([accounts_data[bookId]['qb']['t'] for bookId in books]),
                quote_decimals
            )
            total_quote_loan = round(
                sum([accounts_data[bookId]['ql'] for bookId in books]),
                quote_decimals
            )
            total_quote_collateral = round(
                sum([accounts_data[bookId]['qc'] for bookId in books]),
                quote_decimals
            )

            total_daily_volume = {
                role: round(
                    sum([book_volume[role] for book_volume in daily_volumes[agentId].values()]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            average_daily_volume = {
                role: round(
                    total_daily_volume[role] / len(daily_volumes[agentId]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            min_daily_volume = {
                role: min([book_volume[role] for book_volume in daily_volumes[agentId].values()])
                for role in ['total', 'maker', 'taker', 'self']
            }

            activity_factor = (
                sum(validator_data['activity_factors'][agentId].values()) /
                len(validator_data['activity_factors'][agentId])
            )

            sharpe_values = validator_data['sharpe_values'][agentId] if agentId in validator_data['sharpe_values'] else None

            miner_metrics[agentId] = {
                'total_base_balance': total_base_balance,
                'total_base_loan': total_base_loan,
                'total_base_collateral': total_base_collateral,
                'total_quote_balance': total_quote_balance,
                'total_quote_loan': total_quote_loan,
                'total_quote_collateral': total_quote_collateral,
                'total_inventory_value': total_inventory_history[agentId][-1],
                'inventory_value_change': (
                    total_inventory_history[agentId][-1] - total_inventory_history[agentId][-2]
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'pnl': pnl[agentId],
                'pnl_change': (
                    pnl[agentId] - (total_inventory_history[agentId][-2] - total_inventory_history[agentId][0])
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'total_daily_volume': total_daily_volume,
                'average_daily_volume': average_daily_volume,
                'min_daily_volume': min_daily_volume,
                'activity_factor': activity_factor,
                'sharpe': sharpe_values['median'] if sharpe_values else None,
                'sharpe_penalty': sharpe_values.get('penalty') if sharpe_values else None,
                'sharpe_score': sharpe_values.get('score') if sharpe_values else None,
                'activity_weighted_normalized_median': sharpe_values.get('activity_weighted_normalized_median') if sharpe_values else None,
                'unnormalized_score': validator_data['unnormalized_scores'][agentId],
                'score': scores[agentId].item(),
                'placement': placements[agentId].item(),
            }

        result['metrics'] = {
            'miner_metrics': miner_metrics,
            'daily_volumes': daily_volumes,
            'total_inventory_history': total_inventory_history,
            'pnl': pnl,
            'scores': scores.tolist(),
            'placements': placements.tolist(),
        }
    except Exception as ex:
        result['error'] = str(ex)
        result['traceback'] = traceback.format_exc()
    return result


async def report(self: 'Validator') -> None:
    """
    Calculates and publishes metrics related to simulation state, validator and agent performance.

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
    Returns:
        None
    """
    try:
        self.shared_state.reporting = True
        report_step = self.step
        simulation_duration = duration_from_timestamp(self.simulation_timestamp)
        bt.logging.info(f"Publishing Metrics at Step {self.step} ({simulation_duration})...")
        report_start = time.time()
        bt.logging.debug(f"Publishing simulation metrics...")
        start = time.time()

        _set_if_changed(
            self.prometheus_simulation_gauges,
            self.simulation_timestamp,
            self.wallet.hotkey.ss58_address,
            self.config.netuid,
            "timestamp"
        )

        _set_if_changed(
            self.prometheus_simulation_gauges,
            sum(self.step_rates) / len(self.step_rates) if len(self.step_rates) > 0 else 0,
            self.wallet.hotkey.ss58_address,
            self.config.netuid,
            "step_rate"
        )

        has_new_trades = False
        has_new_miner_trades = False

        publish_validator_gauges(self)

        self.prometheus_books.clear()
        bt.logging.debug(f"Simulation metrics published ({time.time()-start:.4f}s).")

        if self.simulation.logDir:
            bt.logging.debug(f"Retrieving fundamental prices...")
            start = time.time()
            self.load_fundamental()
            bt.logging.debug(f"Retrieved fundamental prices ({time.time()-start:.4f}s).")

        bt.logging.debug(f"Publishing book metrics...")
        for bookId, book in self.last_state.books.items():
            await asyncio.sleep(0)
            # --- Book bids ---
            if book['b']:
                start = time.time()
                bid_cumsum = 0
                for i, level in enumerate(book['b']):
                    await asyncio.sleep(0)
                    _set_if_changed(self.prometheus_book_gauges, level['p'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "bid")
                    _set_if_changed(self.prometheus_book_gauges, level['q'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "bid_vol")
                    bid_cumsum += level['q']
                    _set_if_changed(self.prometheus_book_gauges, bid_cumsum,
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "bid_vol_sum")
                    if i == 20: break
            # --- Book asks ---
            if book['a']:
                start = time.time()
                ask_cumsum = 0
                for i, level in enumerate(book['a']):
                    await asyncio.sleep(0)
                    _set_if_changed(self.prometheus_book_gauges, level['p'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "ask")
                    _set_if_changed(self.prometheus_book_gauges, level['q'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "ask_vol")
                    ask_cumsum += level['q']
                    _set_if_changed(self.prometheus_book_gauges, ask_cumsum,
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, i, "ask_vol_sum")
                    if i == 20: break

            bt.logging.debug(f"Book {bookId} levels metrics published ({time.time()-start:.4f}s).")

            # --- Book aggregate metrics ---
            if book['b'] and book['a']:
                start = time.time()
                mid = (book['b'][0]['p'] + book['a'][0]['p']) / 2
                _set_if_changed(self.prometheus_book_gauges, mid,
                    self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "mid")

                def get_price(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['p'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['p'] if len(book['a']) > idx else 0

                def get_vol(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['q'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['q'] if len(book['a']) > idx else 0

                _set_if_changed(self.prometheus_books, 1.0,
                    self.wallet.hotkey.ss58_address, self.config.netuid, self.simulation_timestamp, simulation_duration, bookId,
                    get_price('bid',4), get_vol('bid',4), get_price('bid',3), get_vol('bid',3), get_price('bid',2), get_vol('bid',2),
                    get_price('bid',1), get_vol('bid',1), get_price('bid',0), get_vol('bid',0),
                    get_price('ask',4), get_vol('ask',4), get_price('ask',3), get_vol('ask',3), get_price('ask',2), get_vol('ask',2),
                    get_price('ask',1), get_vol('ask',1), get_price('ask',0), get_vol('ask',0),
                    "books"
                )
                bt.logging.debug(f"Book {bookId} aggregate metrics published ({time.time()-start:.4f}s).")

            # --- Book trade events ---
            if book['e']:
                trades = [event for event in book['e'] if event['y'] == 't']
                if trades:
                    start = time.time()
                    last_trade = trades[-1]
                    if isinstance(self.fundamental_price[0], pd.Series):
                        _set_if_changed(self.prometheus_book_gauges,
                            self.fundamental_price[bookId].iloc[-1],
                            self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "fundamental_price")
                    else:
                        if self.fundamental_price[bookId]:
                            _set_if_changed(self.prometheus_book_gauges,
                                self.fundamental_price[bookId],
                                self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "fundamental_price")
                        else:
                            try:
                                self.prometheus_book_gauges.remove(self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "fundamental_price")
                            except KeyError:
                                pass

                    _set_if_changed(self.prometheus_book_gauges, last_trade['p'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "trade_price")
                    _set_if_changed(self.prometheus_book_gauges, sum([trade['q'] for trade in trades]),
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "trade_volume")
                    _set_if_changed(self.prometheus_book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 0]),
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "trade_buy_volume")
                    _set_if_changed(self.prometheus_book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 1]),
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "trade_sell_volume")

                    has_new_trades = True

                bt.logging.debug(f"Book {bookId} events metrics published ({time.time()-start:.4f}s).")
            
            # --- Book Fees ---
            if self.simulation.fee_policy.fee_type == 'dynamic':
                DISMTR = self.last_state.books[bookId]['mtr']
                DISmakerRate = self.last_state.accounts[0][bookId]['f']['m']
                DIStakerRate = self.last_state.accounts[0][bookId]['f']['t']
                _set_if_changed(self.prometheus_book_gauges, DISmakerRate,
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "dynamic_maker_rate")
                _set_if_changed(self.prometheus_book_gauges, DIStakerRate,
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "dynamic_taker_rate")
                _set_if_changed(self.prometheus_book_gauges, DISMTR,
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, 0, "maker_taker_ratio")

        # --- Trades metrics ---
        if has_new_trades:
            bt.logging.debug(f"Publishing trade metrics...")
            while self.shared_state.rewarding:
                bt.logging.info(f"Waiting for reward calculation to complete before reporting trades...")
                await asyncio.sleep(0.5)
            start = time.time()
            self.prometheus_trades.clear()
            for bookId, trades in self.recent_trades.items():
                await asyncio.sleep(0)
                for trade in trades:
                    await asyncio.sleep(0)
                    _set_if_changed(self.prometheus_trades, 1.0,
                        self.wallet.hotkey.ss58_address, self.config.netuid, trade.timestamp, duration_from_timestamp(trade.timestamp),
                        bookId, trade.taker_agent_id, trade.id, trade.taker_id, trade.taker_agent_id, trade.maker_id, trade.maker_agent_id,
                        trade.maker_fee, trade.taker_fee, trade.price, trade.quantity, trade.side, "trades")

        bt.logging.debug(f"Trade metrics published ({time.time()-start:.4f}s).")

        if not self.last_state.accounts:
            bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
            return
        bt.logging.debug(f"Computing miner metrics in worker process...")
        computation_start = time.time()
        while self.shared_state.rewarding:
            bt.logging.info(f"Waiting for reward calculation to complete before computing metrics...")
            await asyncio.sleep(0.5)

        validator_data = {
            'simulation_timestamp': self.simulation_timestamp,
            'step': self.step,
            'trade_volumes': self.trade_volumes,
            'inventory_history': self.inventory_history,
            'activity_factors': self.activity_factors,
            'sharpe_values': self.sharpe_values,
            'unnormalized_scores': self.unnormalized_scores,
            'scores': {i: score.item() for i, score in enumerate(self.scores)},
            'book_count': self.simulation.book_count,
            'simulation_config': {
                'volumeDecimals': self.simulation.volumeDecimals,
                'baseDecimals': self.simulation.baseDecimals,
                'quoteDecimals': self.simulation.quoteDecimals,
            }
        }

        state_data = {
            'accounts': self.last_state.accounts,
            'books': self.last_state.books,
            'notices': self.last_state.notices,
        }

        result = await asyncio.get_running_loop().run_in_executor(
            self.report_executor, report_worker, validator_data, state_data
        )

        if result['error']:
            bt.logging.error(f"Error in report worker: {result['error']}")
            bt.logging.debug(f"Traceback: {result.get('traceback', 'N/A')}")
            return

        bt.logging.debug(f"Miner metrics computed ({time.time()-computation_start:.4f}s).")

        metrics = result['metrics']
        miner_metrics = metrics['miner_metrics']
        daily_volumes = metrics['daily_volumes']

        bt.logging.debug(f"Publishing accounts metrics...")
        start = time.time()

        for agentId, accounts in self.last_state.accounts.items():
            await asyncio.sleep(0)
            initial_balance_publish_status = {bookId: False for bookId in range(self.simulation.book_count)}
            for bookId, account in accounts.items():
                await asyncio.sleep(0)
                if self.initial_balances[agentId][bookId]['BASE'] is not None and not self.initial_balances_published[agentId]:
                    _set_if_changed(self.prometheus_agent_gauges, self.initial_balances[agentId][bookId]['BASE'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_balance_initial")
                    _set_if_changed(self.prometheus_agent_gauges, self.initial_balances[agentId][bookId]['QUOTE'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_balance_initial")
                    _set_if_changed(self.prometheus_agent_gauges, self.initial_balances[agentId][bookId]['WEALTH'],
                        self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "wealth_initial")
                    initial_balance_publish_status[bookId] = True
            if all(initial_balance_publish_status.values()):
                self.initial_balances_published[agentId] = True

            if agentId < 0 or len(self.inventory_history[agentId]) < 3:
                continue

            start_inv = [i for i in list(self.inventory_history[agentId].values()) if len(i) > bookId][0]
            last_inv = list(self.inventory_history[agentId].values())[-1]
            sharpes = self.sharpe_values[agentId]

            for bookId, account in accounts.items():
                await asyncio.sleep(0)
                _set_if_changed(self.prometheus_agent_gauges, account['bb']['t'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_balance_total")
                _set_if_changed(self.prometheus_agent_gauges, account['bb']['f'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_balance_free")
                _set_if_changed(self.prometheus_agent_gauges, account['bb']['r'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_balance_reserved")
                _set_if_changed(self.prometheus_agent_gauges, account['qb']['t'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_balance_total")
                _set_if_changed(self.prometheus_agent_gauges, account['qb']['f'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_balance_free")
                _set_if_changed(self.prometheus_agent_gauges, account['qb']['r'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_balance_reserved")
                _set_if_changed(self.prometheus_agent_gauges, account['bl'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_loan")
                _set_if_changed(self.prometheus_agent_gauges, account['bc'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "base_collateral")
                _set_if_changed(self.prometheus_agent_gauges, account['ql'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_loan")
                _set_if_changed(self.prometheus_agent_gauges, account['qc'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "quote_collateral")
                if account['f']['v']:
                    _set_if_changed(self.prometheus_agent_gauges, account['f']['v'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "fees_traded_volume")
                _set_if_changed(self.prometheus_agent_gauges, account['f']['m'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "fees_maker_rate")
                _set_if_changed(self.prometheus_agent_gauges, account['f']['t'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "fees_taker_rate")
                _set_if_changed(self.prometheus_agent_gauges, last_inv[bookId], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "inventory_value")
                _set_if_changed(self.prometheus_agent_gauges, last_inv[bookId] - start_inv[bookId], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "pnl")
                _set_if_changed(self.prometheus_agent_gauges, daily_volumes[agentId][bookId]['total'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "daily_volume")
                _set_if_changed(self.prometheus_agent_gauges, daily_volumes[agentId][bookId]['maker'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "daily_maker_volume")
                _set_if_changed(self.prometheus_agent_gauges, daily_volumes[agentId][bookId]['taker'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "daily_taker_volume")
                _set_if_changed(self.prometheus_agent_gauges, daily_volumes[agentId][bookId]['self'], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "daily_self_volume")
                _set_if_changed(self.prometheus_agent_gauges, self.activity_factors[agentId][bookId], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "activity_factor")
                if sharpes:
                    _set_if_changed(self.prometheus_agent_gauges, sharpes['books'][bookId], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "sharpe")
                    if 'books_weighted' in sharpes:
                        _set_if_changed(self.prometheus_agent_gauges, sharpes['books_weighted'][bookId], self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "weighted_sharpe")
                else:
                    try:
                        self.prometheus_agent_gauges.remove(self.wallet.hotkey.ss58_address, self.config.netuid, bookId, agentId, "sharpe")
                    except KeyError:
                        pass

        bt.logging.debug(f"Agent book metrics published ({time.time()-start:.4f}s).")

        bt.logging.debug(f"Publishing miner trade metrics...")
        start = time.time()
        for agentId, notices in self.last_state.notices.items():
            if agentId < 0:
                continue
            for notice in notices:
                if notice['y'] in ["EVENT_TRADE", "ET"]:
                    has_new_miner_trades = True
                    break
            if has_new_miner_trades:
                break

        if has_new_miner_trades:
            self.prometheus_miner_trades.clear()
            for uid, book_miner_trades in self.recent_miner_trades.items():
                for bookId, miner_trades in book_miner_trades.items():
                    if len(miner_trades) > 0:
                        await asyncio.sleep(0)
                        last_maker_trade = None
                        last_taker_trade = None
                        for miner_trade, role in self.recent_miner_trades[uid][bookId]:
                            _set_if_changed(self.prometheus_miner_trades, 1.0,
                                self.wallet.hotkey.ss58_address, self.config.netuid,
                                miner_trade.timestamp, duration_from_timestamp(miner_trade.timestamp),
                                miner_trade.bookId, uid, role,
                                miner_trade.price, miner_trade.quantity,
                                miner_trade.side if role == 'taker' else int(not miner_trade.side),
                                miner_trade.makerFee if role == 'maker' else miner_trade.takerFee,
                                "miner_trades"
                            )
                            if role == 'maker':
                                last_maker_trade = miner_trade
                            if role == 'taker':
                                last_taker_trade = miner_trade
                        if last_maker_trade:
                            _set_if_changed(self.prometheus_agent_gauges, last_maker_trade.makerFeeRate, self.wallet.hotkey.ss58_address, self.config.netuid, bookId, uid, "fees_last_maker_rate")
                        if last_taker_trade:
                            _set_if_changed(self.prometheus_agent_gauges, last_taker_trade.takerFeeRate, self.wallet.hotkey.ss58_address, self.config.netuid, bookId, uid, "fees_last_taker_rate")
        self.prometheus_miners.clear()

        for agentId in miner_metrics:
            await asyncio.sleep(0)
            m = miner_metrics[agentId]

            _set_if_changed(self.prometheus_miner_gauges, m['total_base_balance'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_base_balance")
            _set_if_changed(self.prometheus_miner_gauges, m['total_base_loan'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_base_loan")
            _set_if_changed(self.prometheus_miner_gauges, m['total_base_collateral'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_base_collateral")
            _set_if_changed(self.prometheus_miner_gauges, m['total_quote_balance'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_quote_balance")
            _set_if_changed(self.prometheus_miner_gauges, m['total_quote_loan'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_quote_loan")
            _set_if_changed(self.prometheus_miner_gauges, m['total_quote_collateral'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_quote_collateral")
            _set_if_changed(self.prometheus_miner_gauges, m['total_inventory_value'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_inventory_value")
            _set_if_changed(self.prometheus_miner_gauges, m['pnl'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "pnl")

            _set_if_changed(self.prometheus_miner_gauges, m['total_daily_volume']['total'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_daily_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['total_daily_volume']['maker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_daily_maker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['total_daily_volume']['taker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_daily_taker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['total_daily_volume']['self'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "total_daily_self_volume")

            _set_if_changed(self.prometheus_miner_gauges, m['average_daily_volume']['total'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "average_daily_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['average_daily_volume']['maker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "average_daily_maker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['average_daily_volume']['taker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "average_daily_taker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['average_daily_volume']['self'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "average_daily_self_volume")

            _set_if_changed(self.prometheus_miner_gauges, m['min_daily_volume']['total'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "min_daily_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['min_daily_volume']['maker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "min_daily_maker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['min_daily_volume']['taker'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "min_daily_taker_volume")
            _set_if_changed(self.prometheus_miner_gauges, m['min_daily_volume']['self'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "min_daily_self_volume")

            _set_if_changed(self.prometheus_miner_gauges, m['activity_factor'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "activity_factor")

            if m['sharpe'] is not None:
                _set_if_changed(self.prometheus_miner_gauges, m['sharpe'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "sharpe")
                if m['activity_weighted_normalized_median'] is not None:
                    _set_if_changed(self.prometheus_miner_gauges, m['activity_weighted_normalized_median'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "activity_weighted_normalized_median_sharpe")
                if m['sharpe_penalty'] is not None:
                    _set_if_changed(self.prometheus_miner_gauges, m['sharpe_penalty'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "sharpe_penalty")
                if m['sharpe_score'] is not None:
                    _set_if_changed(self.prometheus_miner_gauges, m['sharpe_score'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "sharpe_score")
            else:
                try:
                    self.prometheus_miner_gauges.remove(self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "sharpe")
                except KeyError:
                    pass

            _set_if_changed(self.prometheus_miner_gauges, m['unnormalized_score'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "unnormalized_score")
            _set_if_changed(self.prometheus_miner_gauges, m['score'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "score")
            _set_if_changed(self.prometheus_miner_gauges, m['placement'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "placement")

            _set_if_changed(self.prometheus_miner_gauges, (self.metagraph.trust[agentId] if len(self.metagraph.trust) > agentId else 0.0), self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "trust")
            _set_if_changed(self.prometheus_miner_gauges, (self.metagraph.consensus[agentId] if len(self.metagraph.consensus) > agentId else 0.0), self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "consensus")
            _set_if_changed(self.prometheus_miner_gauges, (self.metagraph.incentive[agentId] if len(self.metagraph.incentive) > agentId else 0.0), self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "incentive")
            _set_if_changed(self.prometheus_miner_gauges, (self.metagraph.emission[agentId] if len(self.metagraph.emission) > agentId else 0.0), self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "emission")

            if self.simulation_timestamp % (self.simulation.publish_interval * 100) == 0:
                _set_if_changed(self.prometheus_miner_gauges, self.miner_stats[agentId]['requests'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "requests")
                _set_if_changed(self.prometheus_miner_gauges, self.miner_stats[agentId]['requests'] - self.miner_stats[agentId]['failures'] - self.miner_stats[agentId]['timeouts'] - self.miner_stats[agentId]['rejections'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "success")
                _set_if_changed(self.prometheus_miner_gauges, self.miner_stats[agentId]['failures'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "failures")
                _set_if_changed(self.prometheus_miner_gauges, self.miner_stats[agentId]['timeouts'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "timeouts")
                _set_if_changed(self.prometheus_miner_gauges, self.miner_stats[agentId]['rejections'], self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "rejections")
                _set_if_changed(self.prometheus_miner_gauges, (sum(self.miner_stats[agentId]['call_time']) / len(self.miner_stats[agentId]['call_time']) if len(self.miner_stats[agentId]['call_time']) > 0 else 0), self.wallet.hotkey.ss58_address, self.config.netuid, agentId, "call_time")
                self.miner_stats[agentId] = {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}

            _set_if_changed_metric(
                self.prometheus_miners,
                1.0,
                wallet=self.wallet.hotkey.ss58_address,
                netuid=self.config.netuid,
                agent_id=agentId,
                timestamp=self.simulation_timestamp,
                timestamp_str=duration_from_timestamp(self.simulation_timestamp),
                placement=m['placement'],
                base_balance=m['total_base_balance'],
                base_loan=m['total_base_loan'],
                base_collateral=m['total_base_collateral'],
                quote_balance=m['total_quote_balance'],
                quote_loan=m['total_quote_loan'],
                quote_collateral=m['total_quote_collateral'],
                inventory_value=m['total_inventory_value'],
                inventory_value_change=m['inventory_value_change'],
                pnl=m['pnl'],
                pnl_change=m['pnl_change'],
                min_daily_volume=m['min_daily_volume']['total'],
                activity_factor=m['activity_factor'],
                sharpe=m['sharpe'],
                sharpe_penalty=m['sharpe_penalty'],
                sharpe_score=m['sharpe_score'],
                unnormalized_score=m['unnormalized_score'],
                score=m['score'],
                miner_gauge_name='miners'
            )

        bt.logging.debug(f"Miner and metagraph metrics published ({time.time()-start:.4f}s).")
        bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
    except Exception as ex:
        self.pagerduty_alert(f"Unable to publish metrics : {ex}", details={"traceback": traceback.format_exc()})
    finally:
        self.shared_state.reporting = False