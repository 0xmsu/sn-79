# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)

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

if __name__ != "__mp_main__":
    import os
    import json
    import signal
    import sys
    import platform
    import time
    import argparse
    import torch
    import traceback
    import xml.etree.ElementTree as ET
    import msgspec
    import math
    import shutil
    import zipfile
    import asyncio
    import posix_ipc
    import mmap
    import msgpack
    import atexit
    import multiprocessing
    from datetime import datetime, timedelta
    from ypyjson import YpyObject

    import bittensor as bt

    import uvicorn
    from typing import Tuple, Dict
    from fastapi import FastAPI, APIRouter
    from fastapi import Request
    import threading
    from threading import Thread, Lock, Event
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

    import psutil
    from git import Repo
    from pathlib import Path

    from taos import __spec_version__
    from taos.common.neurons.validator import BaseValidatorNeuron
    from taos.im.utils import duration_from_timestamp
    from taos.im.utils.save import save_state_worker
    from taos.im.utils.reward import get_inventory_value

    from taos.im.config import add_im_validator_args
    from taos.im.protocol.simulator import SimulatorResponseBatch
    from taos.im.protocol import MarketSimulationStateUpdate, FinanceEventNotification
    from taos.im.protocol.models import MarketSimulationConfig, TradeInfo
    from taos.im.protocol.events import SimulationStartEvent, TradeEvent

    class Validator(BaseValidatorNeuron):
        """
        intelligent market simulation validator implementation.

        The validator is run as a FastAPI client in order to receive messages from the simulator engine for processing and forwarding to miners.
        Metagraph maintenance, weight setting, state persistence and other general bittensor routines are executed in a separate thread.
        The validator also handles publishing of metrics via Prometheus for visualization and analysis, as well as retrieval and recording of seed data for simulation price process generation.
        """

        @classmethod
        def add_args(cls, parser: argparse.ArgumentParser) -> None:
            """
            Add intelligent-markets-specific validator configuration arguments.
            """
            add_im_validator_args(cls, parser)

        async def wait_for_event(self, event: asyncio.Event, wait_process: str, run_process: str):
            if not event.is_set():
                bt.logging.info(f"Waiting for {wait_process} to complete before {run_process}...")
                start_wait = time.time()
                while not event.is_set():
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_wait
                        bt.logging.info(f"Waiting for {wait_process} to complete before {run_process}... ({elapsed:.1f}s)")

                total_wait = time.time() - start_wait
                bt.logging.info(f"Completed {wait_process} after {total_wait:.1f}s, {run_process}...")

        async def _maintain(self) -> None:
            """
            Async wrapper for maintenance that runs sync operations in executor.
            """
            try:
                self.maintaining = True
                await self.wait_for_event(self._query_done_event, "query", "synchronizing")
                bt.logging.info(f"Synchronizing at Step {self.step}...")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.maintenance_executor,
                    self._sync_and_check
                )
                bt.logging.info(f"Synchronized ({time.time()-start:.4f}s)")

            except Exception as ex:
                self.pagerduty_alert(f"Failed to sync: {ex}", details={"trace": traceback.format_exc()})
            finally:
                self.maintaining = False

        def _sync_and_check(self):
            """
            Helper function that runs sync operations.
            """
            self.sync(save_state=False)
            if not check_simulator(self):
                restart_simulator(self)

        def maintain(self) -> None:
            """
            Maintains the metagraph and sets weights.
            """
            if not self.maintaining and self.last_state and self.last_state.timestamp % self.config.scoring.interval == 2_000_000_000:
                bt.logging.debug(f"[MAINT] Scheduling from thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINT] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
                self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._maintain()))

        def monitor(self) -> None:
            while True:
                try:
                    time.sleep(300)
                    bt.logging.info(f"Checking simulator state...")
                    if not check_simulator(self):
                        restart_simulator(self)
                    else:
                        bt.logging.info(f"Simulator online!")
                except Exception as ex:
                    bt.logging.error(f"Failure in simulator monitor : {traceback.format_exc()}")

        def seed(self) -> None:
            from taos.im.validator.seed import seed
            seed(self)

        def update_repo(self, end=False) -> bool:
            """
            Checks for and pulls latest changes from the taos repo.
            If source has changed, the simulator is rebuilt and restarted.
            If config has changed, restart the simulator to activate the new parameterizations.
            If validator source is updated, restart validator process.
            """
            try:
                validator_py_files_changed, simulator_config_changed, simulator_py_files_changed, simulator_cpp_files_changed = check_repo(self)
                remote = self.repo.remotes[self.config.repo.remote]

                if not end:
                    if validator_py_files_changed and not (simulator_cpp_files_changed or simulator_py_files_changed):
                        bt.logging.warning("VALIDATOR LOGIC UPDATED - PULLING AND DEPLOYING.")
                        remote.pull()
                        update_validator(self)
                else:
                    try:
                        remote.pull()
                    except Exception as ex:
                        self.pagerduty_alert(f"Failed to pull changes from repo on simulation end : {ex}")
                    if simulator_cpp_files_changed or simulator_py_files_changed:
                        bt.logging.warning("SIMULATOR SOURCE CHANGED")
                        rebuild_simulator(self)
                    if simulator_config_changed:
                        bt.logging.warning("SIMULATOR CONFIG CHANGED")
                    restart_simulator(self)
                    if validator_py_files_changed:
                        update_validator(self)
                return True
            except Exception as ex:
                self.pagerduty_alert(f"Failed to update repo : {ex}", details={"traceback" : traceback.format_exc()})
                return False

        def _compress_outputs(self,  start=False):
            self.compressing = True
            try:
                if self.simulation.logDir:
                    log_root = Path(self.simulation.logDir).parent
                    for output_dir in log_root.iterdir():
                        if output_dir.is_dir():
                            log_archives = {}
                            log_path = Path(output_dir)
                            for log_file in log_path.iterdir():
                                if log_file.is_file() and log_file.suffix == '.log':
                                    log_period = log_file.name.split('.')[1]
                                    if len(log_period) == 13:
                                        log_end = (int(log_period.split('-')[1][:2]) * 3600 + int(log_period.split('-')[1][2:4]) * 60 + int(log_period.split('-')[1][4:])) * 1_000_000_000
                                    else:
                                        log_end = (int(log_period.split('-')[1][:2]) * 86400 + int(log_period.split('-')[1][2:4]) * 3600 + int(log_period.split('-')[1][4:6]) * 60 + int(log_period.split('-')[1][6:])) * 1_000_000_000
                                    if log_end < self.simulation_timestamp or (start and str(output_dir.resolve()) != self.simulation.logDir):
                                        log_type = log_file.name.split('-')[0]
                                        label = f"{log_type}_{log_period}"
                                        if not label in log_archives:
                                            log_archives[label] = []
                                        log_archives[label].append(log_file)
                            for label, log_files in log_archives.items():
                                archive = log_path / f"{label}.zip"
                                bt.logging.info(f"Compressing {label} files to {archive.name}...")
                                with zipfile.ZipFile(archive, "w" if not archive.exists() else "a", compression=zipfile.ZIP_DEFLATED) as zipf:
                                    for log_file in log_files:
                                        try:
                                            zipf.write(log_file, arcname=Path(log_file).name)
                                            os.remove(log_file)
                                            bt.logging.debug(f"Added {log_file.name} to {archive.name}")
                                        except Exception as ex:
                                            bt.logging.error(f"Failed to add {log_file.name} to {archive.name} : {ex}")
                    if psutil.disk_usage('/').percent > 85:
                        min_retention_date = int((datetime.today() - timedelta(days=7)).strftime("%Y%m%d"))
                        bt.logging.warning(f"Disk usage > 85% - cleaning up old outputs...")
                        for output in sorted(log_root.iterdir(), key=lambda f: f.name[:13]):
                            try:
                                archive_date = int(output.name[:8])
                            except:
                                continue
                            if archive_date < min_retention_date:
                                try:
                                    if output.is_file() and output.name.endswith('.zip'):
                                        output.unlink()
                                    elif output.is_dir():
                                        shutil.rmtree(output)
                                    disk_usage = psutil.disk_usage('/').percent
                                    bt.logging.success(f"Deleted {output.name} ({disk_usage}% disk available).")
                                    if disk_usage <= 85:
                                        break
                                except Exception as ex:
                                    self.pagerduty_alert(f"Failed to remove output {output.name} : {ex}", details={"trace" : traceback.format_exc()})


            except Exception as ex:
                self.pagerduty_alert(f"Failure during output compression : {ex}", details={"trace" : traceback.format_exc()})
            finally:
                self.compressing = False

        def compress_outputs(self, start=False):
            if not self.compressing:
                Thread(target=self._compress_outputs, args=(start,), daemon=True, name=f'compress_{self.step}').start()

        async def _save_state(self) -> bool:
            """
            Asynchronously saves validator and simulation state to disk in a separate process.

            Returns:
                bool: True if save was successful, False otherwise
            """
            if self.shared_state_saving:
                bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
                return False

            self.shared_state_saving = True
            await self.wait_for_event(self._query_done_event, "query", "preparing state data")

            try:
                start = time.time()
                prep_start = time.time()
                bt.logging.debug("Preparing state for saving...")

                simulation_state_data = {
                    "start_time": self.start_time,
                    "start_timestamp": self.start_timestamp,
                    "step_rates": self.step_rates,
                    "initial_balances": self.initial_balances,
                    "recent_trades": {
                        book_id: [t.model_dump(mode="json") for t in trades]
                        for book_id, trades in self.recent_trades.items()
                    },
                    "recent_miner_trades": {
                        uid: {
                            book_id: [[t.model_dump(mode="json"), r] for t, r in trades]
                            for book_id, trades in uid_trades.items()
                        }
                        for uid, uid_trades in self.recent_miner_trades.items()
                    },
                    "pending_notices": self.pending_notices,
                    "simulation.logDir": self.simulation.logDir,
                }

                validator_state_data = {
                    "step": self.step,
                    "simulation_timestamp": self.simulation_timestamp,
                    "hotkeys": self.hotkeys,
                    "scores": [score.item() for score in self.scores],
                    "activity_factors": self.activity_factors,
                    "inventory_history": self.inventory_history,
                    "sharpe_values": self.sharpe_values,
                    "unnormalized_scores": self.unnormalized_scores,
                    "trade_volumes": self.trade_volumes,
                    "deregistered_uids": self.deregistered_uids,
                    "volume_sums": self.volume_sums,
                    "maker_volume_sums": self.maker_volume_sums,
                    "taker_volume_sums": self.taker_volume_sums,
                    "self_volume_sums": self.self_volume_sums,
                }
                bt.logging.info(f"Prepared save data ({time.time() - prep_start}s)")
                await self.wait_for_event(self._query_done_event, "query", "saving state")
                bt.logging.info("Saving state...")
                future_start = time.time()

                future = asyncio.get_running_loop().run_in_executor(
                    self.save_state_executor,
                    save_state_worker,
                    simulation_state_data,
                    validator_state_data,
                    self.simulation_state_file,
                    self.validator_state_file
                )
                while not future.done():
                    await asyncio.sleep(0.1)
                result = future.result()
                bt.logging.info(f"Saved state ({time.time() - future_start}s)")

                if result['success']:
                    bt.logging.success(
                        f"Simulation state saved to {self.simulation_state_file} "
                        f"({result['simulation_save_time']:.4f}s)"
                    )
                    bt.logging.success(
                        f"Validator state saved to {self.validator_state_file} "
                        f"({result['validator_save_time']:.4f}s)"
                    )
                    bt.logging.info(f"Total save time: {result['total_time']:.4f}s | {time.time()-start}s")
                    return True
                else:
                    bt.logging.error(f"Failed to save state: {result['error']}")
                    if result.get('traceback'):
                        bt.logging.debug(result['traceback'])
                    self.pagerduty_alert(
                        f"Failed to save state: {result['error']}",
                        details={"trace": result.get('traceback')}
                    )
                    return False

            except Exception as ex:
                bt.logging.error(f"Error preparing state for save: {ex}")
                bt.logging.debug(traceback.format_exc())
                self.pagerduty_alert(
                    f"Failed to prepare state for save: {ex}",
                    details={"trace": traceback.format_exc()}
                )
                return False
            finally:
                self.shared_state_saving = False

        def save_state(self) -> None:
            """
            Synchronous wrapper for save_state that schedules it on the event loop.
            """
            if not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 4_000_000_000:
                return
            if self.shared_state_saving:
                bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
                return
            bt.logging.debug(f"[SAVE] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[SAVE] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._save_state()))

        def _save_state_sync(self):
            """
            Direct synchronous save without using executor.
            Used as fallback when executors are shut down.
            """
            try:
                bt.logging.info("Saving state (sync)...")

                # Prepare state data
                simulation_state_data = {
                    "start_time": self.start_time,
                    "start_timestamp": self.start_timestamp,
                    "step_rates": self.step_rates,
                    "initial_balances": self.initial_balances,
                    "recent_trades": {
                        book_id: [t.model_dump(mode="json") for t in trades]
                        for book_id, trades in self.recent_trades.items()
                    },
                    "recent_miner_trades": {
                        uid: {
                            book_id: [[t.model_dump(mode="json"), r] for t, r in trades]
                            for book_id, trades in uid_trades.items()
                        }
                        for uid, uid_trades in self.recent_miner_trades.items()
                    },
                    "pending_notices": self.pending_notices,
                    "simulation.logDir": self.simulation.logDir,
                }

                validator_state_data = {
                    "step": self.step,
                    "simulation_timestamp": self.simulation_timestamp,
                    "hotkeys": self.hotkeys,
                    "scores": [score.item() for score in self.scores],
                    "activity_factors": self.activity_factors,
                    "inventory_history": self.inventory_history,
                    "sharpe_values": self.sharpe_values,
                    "unnormalized_scores": self.unnormalized_scores,
                    "trade_volumes": self.trade_volumes,
                    "deregistered_uids": self.deregistered_uids,
                }

                # Call worker function directly (synchronously in main process)
                result = save_state_worker(
                    simulation_state_data,
                    validator_state_data,
                    self.simulation_state_file,
                    self.validator_state_file
                )

                if result['success']:
                    bt.logging.success(
                        f"State saved directly: simulation ({result['simulation_save_time']:.4f}s), "
                        f"validator ({result['validator_save_time']:.4f}s)"
                    )
                else:
                    bt.logging.error(f"Direct save failed: {result['error']}")
            except Exception as ex:
                bt.logging.error(f"Error in direct save: {ex}]\n{traceback.format_exc()}")

        def load_state(self) -> None:
            """Loads the state of the validator from a file."""
            if os.path.exists(self.simulation_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch simulation state file exists - converting to msgpack...")
                pt_simulation_state = torch.load(self.simulation_state_file.replace('.mp', '.pt'), weights_only=False)
                with open(self.simulation_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        {
                            "start_time": pt_simulation_state['start_time'],
                            "start_timestamp": pt_simulation_state['start_timestamp'],
                            "step_rates": pt_simulation_state['step_rates'],
                            "initial_balances": pt_simulation_state['initial_balances'],
                            "recent_trades": {book_id : [t.model_dump(mode='json') for t in book_trades] for book_id, book_trades in pt_simulation_state['recent_trades'].items()},
                            "recent_miner_trades": {uid : {book_id : [[t.model_dump(mode='json'), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in pt_simulation_state['recent_miner_trades'].items()},
                            "pending_notices": pt_simulation_state['pending_notices'],
                            "simulation.logDir": pt_simulation_state['simulation.logDir']
                        }, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.simulation_state_file.replace('.mp', '.pt'), self.simulation_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch simulation state file converted to msgpack at {self.simulation_state_file}")

            if not self.config.neuron.reset and os.path.exists(self.simulation_state_file):
                bt.logging.info(f"Loading simulation state variables from {self.simulation_state_file}...")
                with open(self.simulation_state_file, 'rb') as file:
                    byte_data = file.read()
                simulation_state = msgpack.unpackb(byte_data, use_list=True, strict_map_key=False)
                self.start_time = simulation_state["start_time"]
                self.start_timestamp = simulation_state["start_timestamp"]
                self.step_rates = simulation_state["step_rates"]
                self.pending_notices = simulation_state["pending_notices"]
                self.initial_balances = simulation_state["initial_balances"] if 'initial_balances' in simulation_state else {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                for uid, initial_balances in self.initial_balances.items():
                    if not 'WEALTH' in initial_balances[0]:
                        self.initial_balances[uid] = {bookId : initial_balance | {'WEALTH' : self.simulation.miner_wealth} for bookId, initial_balance in initial_balances.items()}
                self.recent_trades = {book_id : [TradeInfo.model_construct(**t) for t in book_trades] for book_id, book_trades in simulation_state["recent_trades"].items()}
                self.recent_miner_trades = {uid : {book_id : [[TradeEvent.model_construct(**t), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in simulation_state["recent_miner_trades"].items()}  if "recent_miner_trades" in simulation_state else {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.simulation.logDir = simulation_state["simulation.logDir"]
                bt.logging.success(f"Loaded simulation state.")
            else:
                # If no state exists or the neuron.reset flag is set, re-initialize the simulation state
                if self.config.neuron.reset and os.path.exists(self.simulation_state_file):
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous state info at {self.simulation_state_file}.")
                else:
                    bt.logging.info(f"No previous state information at {self.simulation_state_file}, initializing new simulation state.")
                self.pending_notices = {uid : [] for uid in range(self.subnet_info.max_uids)}
                self.initial_balances = {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.recent_trades = {bookId : [] for bookId in range(self.simulation.book_count)}
                self.recent_miner_trades = {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.fundamental_price = {bookId : None for bookId in range(self.simulation.book_count)}

            if os.path.exists(self.validator_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch validator state file exists - converting to msgpack...")
                pt_validator_state = torch.load(self.validator_state_file.replace('.mp', '.pt'), weights_only=False)
                pt_validator_state["scores"] = [score.item() for score in pt_validator_state['scores']]
                with open(self.validator_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        pt_validator_state, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.validator_state_file.replace('.mp', '.pt'), self.validator_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch validator state file converted to msgpack at {self.validator_state_file}")

            if not self.config.neuron.reset and os.path.exists(self.validator_state_file):
                bt.logging.info(f"Loading validator state variables from {self.validator_state_file}...")
                with open(self.validator_state_file, 'rb') as file:
                    byte_data = file.read()
                validator_state = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
                self.step = validator_state["step"]
                self.simulation_timestamp = validator_state["simulation_timestamp"] if "simulation_timestamp" in validator_state else 0
                self.hotkeys = validator_state["hotkeys"]
                self.deregistered_uids = list(validator_state["deregistered_uids"]) if "deregistered_uids" in validator_state else []
                self.scores = torch.tensor(validator_state["scores"])
                self.activity_factors = validator_state["activity_factors"] if "activity_factors" in validator_state else {uid : {bookId : 0.0 for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                if isinstance(self.activity_factors[0], float):
                    self.activity_factors = {uid : {bookId : self.activity_factors[uid] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.inventory_history = validator_state["inventory_history"] if "inventory_history" in validator_state else {uid : {} for uid in range(self.subnet_info.max_uids)}
                for uid in self.inventory_history:
                    for timestamp in self.inventory_history[uid]:
                        if len(self.inventory_history[uid][timestamp]) < self.simulation.book_count:
                            for bookId in range(len(self.inventory_history[uid][timestamp]),self.simulation.book_count):
                                self.inventory_history[uid][timestamp][bookId] = 0.0
                        if len(self.inventory_history[uid][timestamp]) > self.simulation.book_count:
                            self.inventory_history[uid][timestamp] = {k : v for k, v in self.inventory_history[uid][timestamp].items() if k < self.simulation.book_count}
                self.sharpe_values = validator_state["sharpe_values"]
                for uid in self.sharpe_values:
                    if self.sharpe_values[uid] and len(self.sharpe_values[uid]['books']) < self.simulation.book_count:
                        for bookId in range(len(self.sharpe_values[uid]['books']),self.simulation.book_count):
                            self.sharpe_values[uid]['books'][bookId] = 0.0
                            self.sharpe_values[uid]['books_weighted'][bookId] = 0.0
                    if self.sharpe_values[uid] and len(self.sharpe_values[uid]['books']) > self.simulation.book_count:
                        self.sharpe_values[uid]['books'] = {k : v for k, v in self.sharpe_values[uid]['books'].items() if k < self.simulation.book_count}
                        if 'books_weighted' in self.sharpe_values[uid]:
                            self.sharpe_values[uid]['books_weighted'] = {k : v for k, v in self.sharpe_values[uid]['books_weighted'].items() if k < self.simulation.book_count}
                self.unnormalized_scores = validator_state["unnormalized_scores"]
                self.trade_volumes = validator_state["trade_volumes"] if "trade_volumes" in validator_state else {uid : {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                reorg = False
                for uid in self.trade_volumes:
                    for bookId in self.trade_volumes[uid]:
                        if not 'total' in self.trade_volumes[uid][bookId]:
                            if not reorg:
                                bt.logging.info(f"Optimizing miner volume history structures...")
                                reorg = True
                            volumes = {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}}
                            for time, role_volume in self.trade_volumes[uid][bookId].items():
                                sampled_time = math.ceil(time / self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval
                                for role, volume in role_volume.items():
                                    if not sampled_time in volumes[role]:
                                        volumes[role][sampled_time] = 0.0
                                    volumes[role][sampled_time] += volume
                            self.trade_volumes[uid][bookId] = {role : {time : round(volumes[role][time], self.simulation.volumeDecimals) for time in volumes[role]} for role in volumes}
                    if len(self.trade_volumes[uid]) < self.simulation.book_count:
                        for bookId in range(len(self.trade_volumes[uid]),self.simulation.book_count):
                            self.trade_volumes[uid][bookId] = {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}}
                    if len(self.trade_volumes[uid]) > self.simulation.book_count:
                        self.trade_volumes[uid] = {k : v for k, v in self.trade_volumes[uid].items() if k < self.simulation.book_count}
                    if len(self.activity_factors[uid]) < self.simulation.book_count:
                        for bookId in range(len(self.activity_factors[uid]),self.simulation.book_count):
                            self.activity_factors[uid][bookId] = 0.0
                    if len(self.activity_factors[uid]) > self.simulation.book_count:
                        self.activity_factors[uid] = {k : v for k, v in self.activity_factors[uid].items() if k < self.simulation.book_count}
                self.volume_sums = validator_state.get('volume_sums', {})
                self.maker_volume_sums = validator_state.get('maker_volume_sums', {})
                self.taker_volume_sums = validator_state.get('taker_volume_sums', {})
                self.self_volume_sums = validator_state.get('self_volume_sums', {})
                if reorg:
                    self._save_state()
                bt.logging.success(f"Loaded validator state.")
            else:
                # If no state exists or the neuron.reset flag is set, re-initialize the validator state
                if self.config.neuron.reset and os.path.exists(self.validator_state_file):
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous state info at {self.validator_state_file}.")
                else:
                    bt.logging.info(f"No previous state information at {self.validator_state_file}, initializing new simulation state.")
                self.activity_factors = {uid : {bookId : 0.0 for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.inventory_history = {uid : {} for uid in range(self.subnet_info.max_uids)}
                self.sharpe_values = {uid :
                    {
                        'books' : {
                            bookId : 0.0 for bookId in range(self.simulation.book_count)
                        },
                        'books_weighted' : {
                            bookId : 0.0 for bookId in range(self.simulation.book_count)
                        },
                        'total' : 0.0,
                        'average' : 0.0,
                        'median' : 0.0,
                        'normalized_average' : 0.0,
                        'normalized_total' : 0.0,
                        'normalized_median' : 0.0,
                        'activity_weighted_normalized_median' : 0.0,
                        'penalty' : 0.0,
                        'score' : 0.0
                    } for uid in range(self.subnet_info.max_uids)
                }
                self.unnormalized_scores = {uid : 0.0 for uid in range(self.subnet_info.max_uids)}
                self.trade_volumes = {uid : {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}

        def load_simulation_config(self) -> None:
            """
            Reads elements from the config XML to populate the simulation config class object.
            """
            self.xml_config = ET.parse(self.config.simulation.xml_config).getroot()
            self.simulation = MarketSimulationConfig.from_xml(self.xml_config)
            self.validator_state_file = self.config.neuron.full_path + f"/validator.mp"
            self.simulation_state_file = self.config.neuron.full_path + f"/{self.simulation.label()}.mp"
            self.load_state()

        def _setup_signal_handlers(self):
            """Setup handlers for graceful shutdown"""
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                bt.logging.info(f"Received {signal_name}, initiating graceful shutdown...")
                self.cleanup()
                sys.exit(0)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)

        def cleanup_executors(self):
            """Clean up all executors and manager on shutdown"""
            executors = {
                'reward_executor': getattr(self, 'reward_executor', None),
                'report_executor': getattr(self, 'report_executor', None),
                'save_state_executor': getattr(self, 'save_state_executor', None),
                'maintenance_executor': getattr(self, 'maintenance_executor', None),
            }

            for name, executor in executors.items():
                if executor is not None:
                    try:
                        bt.logging.info(f"Shutting down {name}...")
                        executor.shutdown(wait=True, cancel_futures=False)
                        bt.logging.info(f"{name} shut down successfully")
                    except Exception as ex:
                        bt.logging.error(f"Error shutting down {name}: {ex}")

            if hasattr(self, 'manager'):
                try:
                    bt.logging.info("Shutting down multiprocessing manager...")
                    self.manager.shutdown()
                    bt.logging.info("Manager shut down successfully")
                except Exception as ex:
                    bt.logging.error(f"Error shutting down manager: {ex}")

        def cleanup(self):
            """Clean up all resources on shutdown"""
            bt.logging.info("Starting validator cleanup...")
            self._cleanup_done = True
            try:
                self.cleanup_executors()
                self._save_state_sync()
            except Exception as ex:
                traceback.print_exc()
                bt.logging.error(f"Error during cleanup: {ex}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()
            return False

        def __init__(self, config=None) -> None:
            """
            Initialize the intelligent markets simulation validator.
            """
            super(Validator, self).__init__(config=config)
            # Load the simulator config XML file data in order to make context and parameters accessible for reporting and output location.

            if not os.path.exists(self.config.simulation.xml_config):
                raise Exception(f"Simulator config does not exist at {self.config.simulation.xml_config}!")
            self.simulator_config_file = os.path.realpath(Path(self.config.simulation.xml_config))
            # Initialize subnet info and other basic validator/simulation properties
            self.subnet_info = self.subtensor.get_metagraph_info(self.config.netuid)
            self.last_state = None
            self.last_response = None
            self.msgpack_error_counter = 0
            self.simulation_timestamp = 0
            self.reward_weights = {"sharpe" : 1.0}
            self.start_time = None
            self.start_timestamp = None
            self.last_state_time = None
            self.step_rates = []

            self.main_loop = asyncio.new_event_loop()
            self._main_loop_ready = Event()

            self.maintaining = False
            self.compressing = False
            self.querying = False
            self._rewarding = False
            self._saving = False
            self._reporting = False
            self._rewarding_lock = Lock()
            self._saving_lock = Lock()
            self._reporting_lock = Lock()
            self.reward_executor = ProcessPoolExecutor(max_workers=1)
            self.report_executor = ProcessPoolExecutor(max_workers=1)
            self.save_state_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_executor = ThreadPoolExecutor(max_workers=1)
            self._setup_signal_handlers()
            self._cleanup_done = False
            atexit.register(self.cleanup)

            self.initial_balances_published = {uid : False for uid in range(self.subnet_info.max_uids)}
            self.volume_sums = {}
            self.maker_volume_sums = {}
            self.taker_volume_sums = {}
            self.self_volume_sums = {}

            self.load_simulation_config()

            # Add routes for methods receiving input from simulator
            self.router = APIRouter()
            self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
            self.router.add_api_route("/account", self.account, methods=["GET"])

            self.repo_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
            self.repo = Repo(self.repo_path)
            self.update_repo()

            self.miner_stats = {uid : {'requests' : 0, 'timeouts' : 0, 'failures' : 0, 'rejections' : 0, 'call_time' : []} for uid in range(self.subnet_info.max_uids)}
            init_metrics(self)
            publish_info(self)

        @property
        def shared_state_rewarding(self):
            with self._rewarding_lock:
                return self._rewarding

        @shared_state_rewarding.setter
        def shared_state_rewarding(self, value):
            with self._rewarding_lock:
                self._rewarding = value

        @property
        def shared_state_saving(self):
            with self._saving_lock:
                return self._saving

        @shared_state_saving.setter
        def shared_state_saving(self, value):
            with self._saving_lock:
                self._saving = value

        @property
        def shared_state_reporting(self):
            with self._reporting_lock:
                return self._reporting

        @shared_state_reporting.setter
        def shared_state_reporting(self, value):
            with self._reporting_lock:
                self._reporting = value

        def load_fundamental(self):
            if self.simulation.logDir:
                prices = {}
                for block in range(self.simulation.block_count):
                    block_file = os.path.join(self.simulation.logDir, f'fundamental.{block * self.simulation.books_per_block}-{self.simulation.books_per_block * (block + 1) - 1}.csv')
                    fp_line = None
                    book_ids = None
                    for line in open(block_file, 'r').readlines():
                        if not book_ids:
                            book_ids = [int(col) for col in line.split(',') if col != "Timestamp\n"]
                        if line.strip() != '':
                            fp_line = line
                    prices = prices | {book_ids[i] : float(price) for i, price in enumerate(fp_line.strip().split(',')[:-1])}
            else:
                prices = {bookId : None for bookId in range(self.simulation.book_count)}
            self.fundamental_price = prices

        def onStart(self, timestamp, event : SimulationStartEvent) -> None:
            """
            Triggered when start of simulation event is published by simulator.
            Sets the simulation output directory and retrieves any fundamental price values already written.
            """
            self.load_simulation_config()
            self.trade_volumes = {
                uid : {
                    bookId : {
                        role : {
                            prev_time - self.simulation_timestamp : volume for prev_time, volume in self.trade_volumes[uid][bookId][role].items() if prev_time - self.simulation_timestamp < self.simulation_timestamp
                        } for role in self.trade_volumes[uid][bookId]
                    } for bookId in range(self.simulation.book_count)
                } for uid in range(self.subnet_info.max_uids)
            }
            self.inventory_history = {
                uid : {
                    prev_time - self.simulation_timestamp : values for prev_time, values in self.inventory_history[uid].items() if prev_time - self.simulation_timestamp < self.simulation_timestamp
                } for uid in range(self.subnet_info.max_uids)
            }
            self.start_time = time.time()
            self.simulation_timestamp = timestamp
            self.start_timestamp = self.simulation_timestamp
            self.last_state_time = None
            self.step_rates = []
            self.simulation.logDir = event.logDir
            self.compress_outputs(start=True)
            bt.logging.info("-"*40)
            bt.logging.info("SIMULATION STARTED")
            bt.logging.info("-"*40)
            bt.logging.info(f"START TIME: {self.start_time}")
            bt.logging.info(f"TIMESTAMP : {self.start_timestamp}")
            bt.logging.info(f"OUT DIR   : {self.simulation.logDir}")
            bt.logging.info("-"*40)
            self.load_fundamental()
            self.initial_balances = {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : self.simulation.miner_wealth} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
            self.recent_trades = {bookId : [] for bookId in range(self.simulation.book_count)}
            self.recent_miner_trades = {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
            self.save_state()
            publish_info(self)

        def onEnd(self) -> None:
            """
            Triggered when end of simulation event is published by simulator.
            Resets quantities as necessary, updates, rebuilds and launches simulator with the latest configuration.
            """
            bt.logging.info("SIMULATION ENDED")
            self.simulation.logDir = None
            self.fundamental_price = {bookId : None for bookId in range(self.simulation.book_count)}
            self.pending_notices = {uid : [] for uid in range(self.subnet_info.max_uids)}
            self.save_state()
            self.update_repo(end=True)

        def handle_deregistration(self, uid) -> None:
            """
            Triggered on deregistration of a UID.
            Flags the UID as marked for balance reset.
            """
            self.deregistered_uids.append(uid)
            self.scores[uid] = 0.0
            bt.logging.debug(f"UID {uid} Deregistered - Scheduled for reset.")

        def process_resets(self, state : MarketSimulationStateUpdate) -> None:
            """
            Checks for and handles agent reset notices (due to deregistration).
            Zeroes scores and clears relevant internal variables.
            """
            for notice in state.notices[self.uid]:
                if notice['y'] in ["RESPONSE_DISTRIBUTED_RESET_AGENT", "RDRA"] or notice['y'] in ["ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", "ERDRA"]:
                    for reset in notice['r']:
                        if reset['u']:
                            bt.logging.info(f"Agent {reset['a']} Balances Reset! {reset}")
                            if reset['a'] in self.deregistered_uids:
                                self.sharpe_values[reset['a']] = {
                                    'books' : {
                                        bookId : 0.0 for bookId in range(self.simulation.book_count)
                                    },
                                    'books_weighted' : {
                                        bookId : 0.0 for bookId in range(self.simulation.book_count)
                                    },
                                    'total' : 0.0,
                                    'average' : 0.0,
                                    'median' : 0.0,
                                    'normalized_average' : 0.0,
                                    'normalized_total' : 0.0,
                                    'normalized_median' : 0.0,
                                    'activity_weighted_normalized_median' : 0.0,
                                    'penalty' : 0.0,
                                    'score' : 0.0
                                }
                                self.unnormalized_scores[reset['a']] = 0.0
                                self.activity_factors[reset['a']] = {bookId : 0.0 for bookId in range(self.simulation.book_count)}
                                self.inventory_history[reset['a']] = {}
                                self.trade_volumes[reset['a']] = {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)}
                                for book_id in range(self.simulation.book_count):
                                    self.volume_sums[(reset['a'], book_id)] = 0.0
                                    self.maker_volume_sums[(reset['a'], book_id)] = 0.0
                                    self.taker_volume_sums[(reset['a'], book_id)] = 0.0
                                    self.self_volume_sums[(reset['a'], book_id)] = 0.0
                                self.initial_balances[reset['a']] = {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)}
                                self.initial_balances_published[reset['a']] = False
                                self.deregistered_uids.remove(reset['a'])
                                self.miner_stats[reset['a']] = {'requests' : 0, 'timeouts' : 0, 'failures' : 0, 'rejections' : 0, 'call_time' : []}
                                self.recent_miner_trades[reset['a']] = {bookId : [] for bookId in range(self.simulation.book_count)}
                        else:
                            self.pagerduty_alert(f"Failed to Reset Agent {reset['a']} : {reset['m']}")

        async def wait_for(self, check_fn: callable, message: str, interval: float = 0.01):
            """
            Wait for a condition to become False.

            Args:
                check_fn: Callable that returns the condition to check
                message: Log message to display while waiting
                interval: Check interval in seconds
            """
            import time

            if not check_fn():
                return

            start_time = time.time()
            last_log_time = start_time

            bt.logging.info(message)

            while check_fn():
                await asyncio.sleep(interval)

                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_log_time >= 1.0:
                    bt.logging.info(f"{message} (waited {elapsed:.1f}s)")
                    last_log_time = current_time

            total_wait = time.time() - start_time
            bt.logging.debug(f"Wait completed after {total_wait:.1f}s")

        async def _report(self):
            try:
                self.shared_state_reporting = True
                await report(self)
            finally:
                self.shared_state_reporting = False

        def report(self) -> None:
            """
            Publish simulation metrics.
            """
            if self.config.reporting.disabled or not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 0:
                return
            if self.shared_state_reporting:
                bt.logging.warning(f"Skipping reporting at step {self.step} — previous report still running.")
                return
            bt.logging.debug(f"[REPORT] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[REPORT] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._report()))

        async def _compute_compact_volumes(self) -> Dict:
            """
            Compute compact volume data for scoring.

            Returns:
                Dict: Compact volume data with lookback_volume and latest_volume per UID/book
            """
            lookback_threshold = self.simulation_timestamp - (
                self.config.scoring.sharpe.lookback *
                self.simulation.publish_interval
            )

            compact_volumes = {}
            for uid in self.metagraph.uids:
                uid_item = uid.item()
                compact_volumes[uid_item] = {}

                if uid_item in self.trade_volumes:
                    for book_id, book_volume in self.trade_volumes[uid_item].items():
                        total_trades = book_volume['total']
                        if not total_trades:
                            compact_volumes[uid_item][book_id] = {
                                'lookback_volume': 0.0,
                                'latest_volume': 0.0
                            }
                            continue
                        timestamps = total_trades.keys()
                        latest_time = max(timestamps)
                        latest_volume = total_trades[latest_time]
                        lookback_volume = sum(
                            vol for t, vol in total_trades.items()
                            if t >= lookback_threshold
                        )
                        compact_volumes[uid_item][book_id] = {
                            'lookback_volume': lookback_volume,
                            'latest_volume': latest_volume
                        }
                else:
                    for book_id in range(self.simulation.book_count):
                        compact_volumes[uid_item][book_id] = {
                            'lookback_volume': 0.0,
                            'latest_volume': 0.0
                        }
            return compact_volumes

        async def _update_trade_volumes(self, state: MarketSimulationStateUpdate):
            """
            Update trade volumes with proper volume_sums maintenance for all roles.
            """
            total_start = time.time()

            books = state.books
            timestamp = state.timestamp
            accounts = state.accounts
            notices = state.notices

            sampled_timestamp = math.ceil(
                timestamp / self.config.scoring.activity.trade_volume_sampling_interval
            ) * self.config.scoring.activity.trade_volume_sampling_interval

            prune_threshold = timestamp - self.config.scoring.activity.trade_volume_assessment_period
            volume_decimals = self.simulation.volumeDecimals
            for bookId, book in books.items():
                trades = [event for event in book['e'] if event['y'] == 't']
                if trades:
                    recent_trades_book = self.recent_trades[bookId]
                    recent_trades_book.extend([TradeInfo.model_construct(**t) for t in trades])
                    del recent_trades_book[:-25]
            for uid in self.metagraph.uids:
                uid_item = uid.item()
                try:
                    if uid_item not in self.trade_volumes:
                        self.trade_volumes[uid_item] = {
                            book_id: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                            for book_id in range(self.simulation.book_count)
                        }
                    trade_volumes_uid = self.trade_volumes[uid_item]
                    for book_id, role_trades in trade_volumes_uid.items():
                        key = (uid_item, book_id)
                        for role, trades in role_trades.items():
                            if trades:
                                pruned_volume = sum(
                                    v for t, v in trades.items() if t < prune_threshold
                                )
                                if pruned_volume > 0:
                                    if role == 'total':
                                        current_sum = self.volume_sums.get(key, 0.0)
                                        self.volume_sums[key] = round(
                                            max(0.0, current_sum - pruned_volume),
                                            volume_decimals
                                        )
                                    elif role == 'maker':
                                        current_sum = self.maker_volume_sums.get(key, 0.0)
                                        self.maker_volume_sums[key] = round(
                                            max(0.0, current_sum - pruned_volume),
                                            volume_decimals
                                        )
                                    elif role == 'taker':
                                        current_sum = self.taker_volume_sums.get(key, 0.0)
                                        self.taker_volume_sums[key] = round(
                                            max(0.0, current_sum - pruned_volume),
                                            volume_decimals
                                        )
                                    elif role == 'self':
                                        current_sum = self.self_volume_sums.get(key, 0.0)
                                        self.self_volume_sums[key] = round(
                                            max(0.0, current_sum - pruned_volume),
                                            volume_decimals
                                        )
                                trade_volumes_uid[book_id][role] = {
                                    t: v for t, v in trades.items() if t >= prune_threshold
                                }
                    for book_id in range(self.simulation.book_count):
                        if book_id not in trade_volumes_uid:
                            trade_volumes_uid[book_id] = {
                                'total': {}, 'maker': {}, 'taker': {}, 'self': {}
                            }
                        book_trade_volumes = trade_volumes_uid[book_id]
                        if sampled_timestamp not in book_trade_volumes['total']:
                            book_trade_volumes['total'][sampled_timestamp] = 0.0
                            book_trade_volumes['maker'][sampled_timestamp] = 0.0
                            book_trade_volumes['taker'][sampled_timestamp] = 0.0
                            book_trade_volumes['self'][sampled_timestamp] = 0.0
                    if uid_item in notices:
                        trades = [notice for notice in notices[uid_item] if notice['y'] in ['EVENT_TRADE', "ET"]]
                        if trades:
                            recent_miner_trades_uid = self.recent_miner_trades[uid_item]
                            volume_deltas = {
                                'total': {},
                                'maker': {},
                                'taker': {},
                                'self': {}
                            }
                            for trade in trades:
                                is_maker = trade['Ma'] == uid_item
                                is_taker = trade['Ta'] == uid_item
                                book_id = trade['b']

                                if is_maker:
                                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "maker"])
                                if is_taker:
                                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "taker"])
                                recent_miner_trades_uid[book_id] = recent_miner_trades_uid[book_id][-5:]

                                book_volumes = trade_volumes_uid[book_id]
                                trade_value = round(trade['q'] * trade['p'], volume_decimals)

                                book_volumes['total'][sampled_timestamp] = round(
                                    book_volumes['total'][sampled_timestamp] + trade_value,
                                    volume_decimals
                                )

                                if book_id not in volume_deltas['total']:
                                    volume_deltas['total'][book_id] = 0.0
                                volume_deltas['total'][book_id] += trade_value

                                if trade['Ma'] == trade['Ta']:
                                    book_volumes['self'][sampled_timestamp] = round(
                                        book_volumes['self'][sampled_timestamp] + trade_value,
                                        volume_decimals
                                    )
                                    if book_id not in volume_deltas['self']:
                                        volume_deltas['self'][book_id] = 0.0
                                    volume_deltas['self'][book_id] += trade_value

                                elif is_maker:
                                    book_volumes['maker'][sampled_timestamp] = round(
                                        book_volumes['maker'][sampled_timestamp] + trade_value,
                                        volume_decimals
                                    )
                                    if book_id not in volume_deltas['maker']:
                                        volume_deltas['maker'][book_id] = 0.0
                                    volume_deltas['maker'][book_id] += trade_value

                                elif is_taker:
                                    book_volumes['taker'][sampled_timestamp] = round(
                                        book_volumes['taker'][sampled_timestamp] + trade_value,
                                        volume_decimals
                                    )
                                    if book_id not in volume_deltas['taker']:
                                        volume_deltas['taker'][book_id] = 0.0
                                    volume_deltas['taker'][book_id] += trade_value

                            for book_id, delta in volume_deltas['total'].items():
                                key = (uid_item, book_id)
                                current_sum = self.volume_sums.get(key, 0.0)
                                self.volume_sums[key] = round(
                                    current_sum + delta,
                                    volume_decimals
                                )

                            for book_id, delta in volume_deltas['maker'].items():
                                key = (uid_item, book_id)
                                current_sum = self.maker_volume_sums.get(key, 0.0)
                                self.maker_volume_sums[key] = round(
                                    current_sum + delta,
                                    volume_decimals
                                )

                            for book_id, delta in volume_deltas['taker'].items():
                                key = (uid_item, book_id)
                                current_sum = self.taker_volume_sums.get(key, 0.0)
                                self.taker_volume_sums[key] = round(
                                    current_sum + delta,
                                    volume_decimals
                                )

                            for book_id, delta in volume_deltas['self'].items():
                                key = (uid_item, book_id)
                                current_sum = self.self_volume_sums.get(key, 0.0)
                                self.self_volume_sums[key] = round(
                                    current_sum + delta,
                                    volume_decimals
                                )

                    if uid_item in accounts:
                        initial_balances_uid = self.initial_balances[uid_item]
                        accounts_uid = accounts[uid_item]

                        for bookId, account in accounts_uid.items():
                            initial_balance_book = initial_balances_uid[bookId]
                            if initial_balance_book['BASE'] is None:
                                initial_balance_book['BASE'] = account['bb']['t']
                            if initial_balance_book['QUOTE'] is None:
                                initial_balance_book['QUOTE'] = account['qb']['t']
                            if initial_balance_book['WEALTH'] is None:
                                initial_balance_book['WEALTH'] = get_inventory_value(account, books[bookId])

                        self.inventory_history[uid_item][timestamp] = {
                            book_id: get_inventory_value(accounts_uid[book_id], book) - initial_balances_uid[book_id]['WEALTH']
                            for book_id, book in books.items()
                        }
                    else:
                        self.inventory_history[uid_item][timestamp] = {book_id: 0.0 for book_id in books}

                    inventory_hist = self.inventory_history[uid_item]
                    if len(inventory_hist) > self.config.scoring.sharpe.lookback:
                        timestamps_to_keep = sorted(inventory_hist.keys())[-self.config.scoring.sharpe.lookback:]
                        self.inventory_history[uid_item] = {
                            ts: inventory_hist[ts] for ts in timestamps_to_keep
                        }
                except Exception as ex:
                    bt.logging.error(f"Failed to update trade data for UID {uid_item}: {ex}")

            total_time = time.time() - total_start
            bt.logging.debug(f"[UPDATE_VOLUMES] Total: {total_time:.4f}s")
            await asyncio.sleep(0)

        async def _reward(self, state : MarketSimulationStateUpdate):
            """
            Calculate and apply rewards for the given simulation state.
            """
            if not hasattr(self, "_reward_lock"):
                self._reward_lock = asyncio.Lock()

            start_wait = time.time()
            async with self._reward_lock:
                waited = time.time() - start_wait
                if waited > 0:
                    bt.logging.debug(f"Acquired reward lock after waiting {waited:.3f}s")
                await asyncio.sleep(0)
                await self.wait_for_event(self._query_done_event, "query", "rewarding")

                self.shared_state_rewarding = True
                await asyncio.sleep(0)

                timestamp = state.timestamp
                duration = duration_from_timestamp(timestamp)
                bt.logging.info(f"Starting reward calculation for step {self.step}...")
                start = time.time()
                await asyncio.sleep(0)

                try:
                    bt.logging.debug("[REWARD] Updating trade volumes...")
                    update_start = time.time()
                    await self._update_trade_volumes(state)
                    bt.logging.debug(f"[REWARD] Trade volumes updated in {time.time()-update_start:.4f}s")
                    if timestamp % self.config.scoring.interval != 0:
                        bt.logging.info(f"Agent Scores Data Updated for {duration} ({time.time()-start:.4f}s)")
                        return
                    bt.logging.debug("[REWARD] Converting inventory history...")
                    await self.wait_for_event(self._query_done_event, "query", "converting inventory history")
                    convert_start = time.time()
                    inventory_compact = {}

                    total_timestamps = 0
                    for uid in self.metagraph.uids:
                        uid_item = uid.item()
                        if uid_item in self.inventory_history and len(self.inventory_history[uid_item]) > 0:
                            hist = self.inventory_history[uid_item]
                            lookback = min(self.config.scoring.sharpe.lookback, len(hist))
                            sorted_timestamps = sorted(hist.keys())[-lookback:]
                            inventory_compact[uid_item] = {ts: hist[ts] for ts in sorted_timestamps}
                            total_timestamps += len(sorted_timestamps)
                        else:
                            inventory_compact[uid_item] = {}

                    bt.logging.debug(f"[REWARD] Converted inventory history in {time.time()-convert_start:.4f}s")
                    await self.wait_for_event(self._query_done_event, "query", "computing compact volumes")
                    compact_start = time.time()
                    compact_volumes = await self._compute_compact_volumes()
                    bt.logging.debug(f"[REWARD] Computed compact volumes in {time.time()-compact_start:.4f}s")

                    prep_start = time.time()
                    validator_data = {
                        'sharpe_values': self.sharpe_values,
                        'activity_factors': self.activity_factors,
                        'compact_volumes': compact_volumes,
                        'inventory_history': inventory_compact,
                        'config': {
                            'scoring': {
                                'sharpe': {
                                    'normalization_min': self.config.scoring.sharpe.normalization_min,
                                    'normalization_max': self.config.scoring.sharpe.normalization_max,
                                    'lookback': self.config.scoring.sharpe.lookback,
                                    'min_lookback': self.config.scoring.sharpe.min_lookback,
                                    'parallel_workers': self.config.scoring.sharpe.parallel_workers if self.config.scoring.sharpe.parallel_workers > 0 else multiprocessing.cpu_count() // 2,
                                },
                                'activity': {
                                    'capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                                    'trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                                    'trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
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
                        },
                        'reward_weights': self.reward_weights,
                        'simulation_timestamp': self.simulation_timestamp,
                        'uids': [uid.item() for uid in self.metagraph.uids],
                        'deregistered_uids': self.deregistered_uids,
                        'device': self.device,
                    }
                    prep_time = time.time() - prep_start
                    bt.logging.debug(f"[REWARD] Prepared validator_data in {prep_time:.4f}s")

                    await asyncio.sleep(0)
                    await self.wait_for_event(self._query_done_event, "query", "submitting reward task to executor")

                    bt.logging.debug("[REWARD] Submitting to executor...")
                    submit_start = time.time()

                    loop = asyncio.get_running_loop()
                    future = loop.run_in_executor(self.reward_executor, get_rewards, validator_data)

                    submit_time = time.time() - submit_start
                    bt.logging.debug(f"[REWARD] Submitted to executor in {submit_time:.4f}s")

                    wait_start = time.time()
                    while not future.done():
                        await asyncio.sleep(0.1)
                    wait_time = time.time() - wait_start
                    bt.logging.debug(f"[REWARD] Executor finished after {wait_time:.4f}s")
                    result_start = time.time()
                    rewards, updated_data = future.result()
                    bt.logging.debug(f"[REWARD] Retrieved result in {time.time() - result_start:.4f}s")

                    self.sharpe_values = updated_data.get('sharpe_values', self.sharpe_values)
                    self.activity_factors = updated_data.get('activity_factors', self.activity_factors)
                    self.simulation_timestamp = updated_data.get('simulation_timestamp', self.simulation_timestamp)

                    bt.logging.debug(f"Agent Rewards Recalculated for {duration} ({time.time()-start:.4f}s):\n{rewards}")
                    await self.wait_for_event(self._query_done_event, "query", "updating scores")
                    self.update_scores(rewards, self.metagraph.uids)
                    bt.logging.info(f"Agent Scores Updated for {duration} ({time.time()-start:.4f}s)")

                except Exception as ex:
                    bt.logging.error(f"Rewarding failed: {ex}\n{traceback.format_exc()}")
                finally:
                    self.shared_state_rewarding = False
                    await asyncio.sleep(0)
                    bt.logging.debug(f"Completed rewarding (TOTAL {time.time()-start_wait:.4f}s).")
            await asyncio.sleep(0)

        def reward(self, state : MarketSimulationStateUpdate) -> None:
            """
            Update agent rewards and recalculate scores.
            """
            bt.logging.debug(f"[REWARD] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[REWARD] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._reward(state)))

        async def handle_state(self, message : dict, state : MarketSimulationStateUpdate, receive_start : int) -> dict:
            # Every 1H of simulation time, check if there are any changes to the validator - if updates exist, pull them and restart.
            if self.simulation_timestamp % 3600_000_000_000 == 0 and self.simulation_timestamp != 0:
                bt.logging.info("Checking for validator updates...")
                self.update_repo()
            state.version = __spec_version__
            start = time.time()
            for uid, accounts in state.accounts.items():
                for book_id in accounts:
                    state.accounts[uid][book_id]['v'] = self.volume_sums.get((uid, book_id), 0.0)
            bt.logging.info(f"Volumes added to state ({time.time()-start:.4f}s).")

            # Update variables
            if not self.start_time:
                self.start_time = time.time()
                self.start_timestamp = state.timestamp
            if self.simulation.logDir != message['logDir']:
                bt.logging.info(f"Simulation log directory changed : {self.simulation.logDir} -> {message['logDir']}")
                self.simulation.logDir = message['logDir']
            self.simulation_timestamp = state.timestamp
            self.step_rates.append((state.timestamp - (self.last_state.timestamp if self.last_state else self.start_timestamp)) / (time.time() - (self.last_state_time if self.last_state_time else self.start_time)))
            self.last_state = state
            if self.simulation:
                state.config = self.simulation.model_copy()
                state.config.simulation_id = os.path.basename(state.config.logDir)[:13]
                state.config.logDir = None
            self.step += 1

            if self.simulation_timestamp % self.simulation.log_window == self.simulation.publish_interval:
                self.compress_outputs()

            # Log received state data
            bt.logging.info(f"STATE UPDATE RECEIVED | VALIDATOR STEP : {self.step} | TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})")
            if self.config.logging.debug or self.config.logging.trace:
                debug_text = ''
                for bookId, book in state.books.items():
                    debug_text += '-' * 50 + "\n"
                    debug_text += f"BOOK {bookId}" + "\n"
                    if book['b'] and book['a']:
                        debug_text += ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in reversed(book['b'][:5])]) + '||' + ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in book['a'][:5]]) + "\n"
                    else:
                        debug_text += "EMPTY" + "\n"
                bt.logging.debug("\n" + debug_text.strip("\n"))

            # Process deregistration notices
            self.process_resets(state)
            # Forward state synapse to miners, populate response data to simulator object and serialize for returning to simulator.
            start = time.time()
            response = SimulatorResponseBatch(await forward(self, state))
            bt.logging.debug(f"Gathered Response Batch ({time.time()-start}s)")
            start = time.time()
            response = response.serialize()
            bt.logging.debug(f"Serialized Response Batch ({time.time()-start}s)")
            # Log response data, start state serialization and reporting threads, and return miner instructions to the simulator
            if len(response['responses']) > 0:
                bt.logging.trace(f"RESPONSE : {response}")
            bt.logging.info(f"RATE : {(self.step_rates[-1] if self.step_rates != [] else 0) / 1e9:.2f} STEPS/s | AVG : {(sum(self.step_rates) / len(self.step_rates) / 1e9 if self.step_rates != [] else 0):.2f}  STEPS/s")
            self.step_rates = self.step_rates[-10000:]
            self.last_state_time = time.time()

            # Calculate latest rewards, update miner scores, save state and publish metrics
            self.maintain()
            self.reward(state)
            self.save_state()
            self.report()
            await asyncio.sleep(0)
            bt.logging.info(f"State update handled ({time.time()-receive_start}s)")

            return response

        async def _listen(self):
            def receive(mq_req: posix_ipc.MessageQueue) -> dict:
                msg, priority = mq_req.receive()
                receive_start = time.time()
                bt.logging.info(f"Received state update from simulator (msgpack)")
                byte_size_req = int.from_bytes(msg, byteorder="little")
                shm_req = posix_ipc.SharedMemory("/state")
                start = time.time()
                packed_data = None
                for attempt in range(1, 6):
                    try:
                        with mmap.mmap(shm_req.fd, byte_size_req, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
                            packed_data = mm.read(byte_size_req)
                        bt.logging.info(f"Read state update ({time.time() - start:.4f}s)")
                        break
                    except Exception as ex:
                        if attempt < 5:
                            bt.logging.error(f"mmap read failed (attempt {attempt}/5): {ex}")
                            time.sleep(0.005)
                        else:
                            bt.logging.error(f"mmap read failed on all 5 attempts: {ex}")
                            self.pagerduty_alert(f"Failed to mmap read after 5 attempts : {ex}", details={"trace": traceback.format_exc()})
                            raise ex
                    finally:
                        if packed_data is not None or attempt >= 5:
                            shm_req.close_fd()
                bt.logging.info(f"Retrieved State Update ({time.time() - receive_start}s)")
                start = time.time()
                result = None
                for attempt in range(1, 6):
                    try:
                        result = msgpack.unpackb(packed_data, raw=False, use_list=True, strict_map_key=False)
                        bt.logging.info(f"Unpacked state update ({time.time() - start:.4f}s)")
                        break
                    except Exception as ex:
                        if attempt < 5:
                            bt.logging.error(f"Msgpack unpack failed (attempt {attempt}/5): {ex}")
                            time.sleep(0.005)
                        else:
                            bt.logging.error(f"Msgpack unpack failed on all 5 attempts: {ex}")
                            self.pagerduty_alert(f"Failed to unpack simulator state after 5 attempts : {ex}", details={"trace": traceback.format_exc()})
                            raise ex
                return result, receive_start

            def respond(response: dict) -> dict:
                self.last_response = response
                packed_res = msgpack.packb(response, use_bin_type=True)
                byte_size_res = len(packed_res)
                mq_res = posix_ipc.MessageQueue("/taosim-res", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
                shm_res = posix_ipc.SharedMemory("/responses", flags=posix_ipc.O_CREAT, size=byte_size_res)
                with mmap.mmap(shm_res.fd, byte_size_res, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ) as mm:
                    shm_res.close_fd()
                    mm.write(packed_res)
                mq_res.send(byte_size_res.to_bytes(8, byteorder="little"))
                mq_res.close()

            mq_req = posix_ipc.MessageQueue("/taosim-req", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
            try:
                while True:
                    response = {"responses": []}
                    try:
                        loop = asyncio.get_event_loop()
                        t1 = time.time()
                        bt.logging.debug(f"[LISTEN] Starting receive at {t1:.3f}")
                        message, receive_start = await loop.run_in_executor(None, receive, mq_req)
                        t2 = time.time()
                        bt.logging.debug(f"[LISTEN] Received message in {t2-t1:.4f}s")
                        state = MarketSimulationStateUpdate.parse_dict(message)
                        t3 = time.time()
                        bt.logging.debug(f"[LISTEN] Parsed state in {t3-t2:.4f}s")
                        response = await self.handle_state(message, state, receive_start)
                        t4 = time.time()
                        bt.logging.debug(f"[LISTEN] handle_state completed in {t4-t3:.4f}s")
                    except Exception as ex:
                        traceback.print_exc()
                        self.pagerduty_alert(f"Exception in posix listener loop : {ex}", details={"trace": traceback.format_exc()})
                    finally:
                        t5 = time.time()
                        bt.logging.debug(f"[LISTEN] Starting respond at {t5:.3f}")
                        await loop.run_in_executor(None, respond, response)
                        t6 = time.time()
                        bt.logging.debug(f"[LISTEN] Respond completed in {t6-t5:.4f}s")
                        bt.logging.debug(f"[LISTEN] Total loop iteration: {t6-t1:.4f}s")
            finally:
                mq_req.close()

        def listen(self):
            """Synchronous wrapper for the asynchronous _listen method."""
            try:
                asyncio.run(self._listen())
            except KeyboardInterrupt:
                print("Listening stopped by user.")

        async def orderbook(self, request : Request) -> dict:
            """
            The route method which receives and processes simulation state updates received from the simulator.
            """
            bt.logging.info("Received state update from simulator.")
            global_start = time.time()
            start = time.time()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            bt.logging.info(f"Retrieved request body ({time.time()-start:.4f}s).")
            if body[-3:].decode() != "]}}":
                raise Exception(f"Incomplete JSON!")
            message = YpyObject(body, 1)
            bt.logging.info(f"Constructed YpyObject ({time.time()-start:.4f}s).")
            state = MarketSimulationStateUpdate.from_ypy(message) # Populate synapse class from request data
            bt.logging.info(f"Synapse populated ({time.time()-start:.4f}s).")
            del body

            response = await self.handle_state(message, state)

            bt.logging.info(f"State update processed ({time.time()-global_start}s)")
            return response

        async def account(self, request : Request) -> None:
            """
            The route method which receives event notification messages from the simulator.
            This method is currently used only to enable the simulation start message to be immediately propagated to the validator.
            Other events are instead recorded to the simulation state object.
            """
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            batch = msgspec.json.decode(body)
            bt.logging.info(f"NOTICE : {batch}")
            notices = []
            ended = False
            for message in batch['messages']:
                if message['type'] == 'EVENT_SIMULATION_START':
                    self.onStart(message['timestamp'], FinanceEventNotification.from_json(message).event)
                    continue
                elif message['type'] == 'EVENT_SIMULATION_END':
                    ended = True
                elif message['type'] == 'RESPONSES_ERROR_REPORT':
                    dump_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.responses.json"
                    with open(dump_file, "w") as f:
                        json.dump(self.last_response, f, indent=4)
                    error_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.error.json"
                    with open(error_file, "w") as f:
                        json.dump(message, f, indent=4)
                    self.msgpack_error_counter += len(message) - 3
                    if self.msgpack_error_counter < 10:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - continuing.", details=message)
                        return { "continue": True }
                    else:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - terminating simulation.", details=message)
                        return { "continue": False }
                notice = FinanceEventNotification.from_json(message)
                if not notice:
                    bt.logging.error(f"Unrecognized notification : {message}")
                else:
                    notices.append(notice)
            await notify(self, notices) # This method forwards the event notifications to the related miners.
            if ended:
                self.onEnd()

# The main method which runs the validator
if __name__ == "__main__":
    from taos.im.validator.update import check_repo, update_validator, check_simulator, rebuild_simulator, restart_simulator
    from taos.im.validator.forward import forward, notify
    from taos.im.validator.report import report, publish_info, init_metrics
    from taos.im.validator.reward import get_rewards

    if float(platform.freedesktop_os_release()['VERSION_ID']) < 22.04:
        raise Exception(f"taos validator requires Ubuntu >= 22.04!")

    bt.logging.info("Initializing validator...")
    app = FastAPI()
    validator = Validator()
    try:
        app.include_router(validator.router)

        bt.logging.info("Starting background threads...")
        threads = []
        for name, target in [('Seed', validator.seed), ('Monitor', validator.monitor), ('Listen', validator.listen)]:
            try:
                bt.logging.info(f"Starting {name} thread...")
                thread = Thread(target=target, daemon=True, name=name)
                thread.start()
                threads.append(thread)
            except Exception as ex:
                validator.pagerduty_alert(f"Exception starting {name} thread: {ex}", details={"trace" : traceback.format_exc()})
                raise

        time.sleep(1)
        for thread in threads:
            if not thread.is_alive():
                validator.pagerduty_alert(f"Failed to start {thread.name} thread!")
                raise RuntimeError(f"Thread '{thread.name}' failed to start")

        bt.logging.info(f"All threads running. Starting FastAPI server and main event loop...")

        def run_main_loop():
            """Run the pre-created main event loop."""
            async def keep_alive():
                bt.logging.info(f"Main event loop started for background tasks")
                bt.logging.debug(f"[MAINLOOP] Thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINLOOP] Loop: {validator.main_loop}")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    bt.logging.info("Main event loop stopping...")
            loop = validator.main_loop
            asyncio.set_event_loop(loop)
            validator._main_loop_ready.set()
            bt.logging.debug(f"[MAINLOOP] Running loop: {loop}")
            try:
                loop.run_until_complete(keep_alive())
            finally:
                loop.close()

        main_loop_thread = Thread(target=run_main_loop, daemon=True, name='main')
        main_loop_thread.start()
        threads.append(main_loop_thread)
        time.sleep(0.5)
        bt.logging.info(f"Starting FastAPI server on port {validator.config.port}...")
        uvicorn.run(app, host="0.0.0.0", port=validator.config.port)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received")
    except Exception as ex:
        bt.logging.error(f"Fatal error: {ex}")
        bt.logging.debug(traceback.format_exc())
        sys.exit(1)