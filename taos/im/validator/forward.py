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
import bittensor as bt
import uvloop
import asyncio
import aiohttp
import multiprocessing
from typing import List

from taos.im.neurons.validator import Validator
from taos.im.protocol import FinanceAgentResponse, FinanceEventNotification, MarketSimulationStateUpdate
from taos.im.protocol.instructions import *
from taos.im.validator.reward import set_delays
from taos.im.utils.compress import compress, batch_compress

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

class DendriteManager:
    @staticmethod
    def configure_session(validator):
        if not validator.dendrite._session or validator.dendrite._session.closed:
            connector = aiohttp.TCPConnector(
                ssl=False,
                limit=0,
                limit_per_host=0,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=validator.config.neuron.timeout,
                connect=1.0,
                sock_read=1.0,
                sock_connect=1.0,
            )

            validator.dendrite._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                skip_auto_headers={'User-Agent'},
            )
            bt.logging.debug("Created new aiohttp session")
        else:
            bt.logging.debug("Reusing existing aiohttp session")

def validate_responses(self: Validator, synapses: dict[int, MarketSimulationStateUpdate]) -> None:
    """
    Checks responses from miners for any attempts at invalid actions, and enforces limits on instruction counts

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
        synapses (dict[int, MarketSimulationStateUpdate]): The synapses with attached agent responses to be validated.
    Returns:
        None
    """
    total_responses = 0
    total_instructions = 0
    success = 0
    timeouts = 0
    failures = 0
    volume_cap = round(self.config.scoring.activity.capital_turnover_cap * self.simulation.miner_wealth, self.simulation.volumeDecimals)
    for uid, synapse in synapses.items():
        valid_instructions = []
        if synapse.is_timeout:
            timeouts += 1
            continue
        elif synapse.is_failure:
            failures += 1
            continue
        elif not synapse.is_success:
            failures += 1
            bt.logging.warning(f"UID {uid} invalid state (not success/timeout/failure): {synapse.dendrite.status_message}")
            continue
        success += 1
        if synapse.is_success:
            synapse.decompress()
        if synapse.response:
            if synapse.compressed:
                bt.logging.warning(f"Failed to decompress response for {uid}!")
                synapse.response = None
                continue
            # If agents attempt to submit instructions for agent IDs other than their own, ignore these responses
            if synapse.response.agent_id != uid:
                bt.logging.warning(f"Invalid response submitted by agent {uid} (Mismatched Agent Ids) : {synapse.response}")
                synapse.response = None
                continue
            miner_volumes = {}
            for book_id in range(self.simulation.book_count):
                cache_key = (uid, book_id)
                miner_volumes[book_id] = self.volume_sums.get(cache_key, 0.0)            
            for instruction in synapse.response.instructions:
                try:
                    if instruction.agentId != uid or instruction.type == 'RESET_AGENT':
                        bt.logging.warning(f"Invalid instruction submitted by agent {uid} (Mismatched Agent Ids) : {instruction}")
                        valid_instructions = []
                        break
                    if instruction.bookId >= self.simulation.book_count:
                        bt.logging.warning(f"Invalid instruction submitted by agent {uid} (Invalid Book Id {instruction.bookId}) : {instruction}")
                        continue
                    # If a miner exceeds `capital_turnover_cap` times their initial wealth in trading volume over a single `trade_volume_assessment_period`, they are restricted from placing additional orders.
                    # Only cancellations may be submitted and processed by the miner until their volume on the specified book in the previous period is below the cap.
                    if miner_volumes[instruction.bookId] >= volume_cap and instruction.type != "CANCEL_ORDERS":
                        bt.logging.debug(f"Agent {uid} has reached their volume cap on book {instruction.bookId} : Traded {miner_volumes[instruction.bookId]} / {volume_cap}.")
                        continue
                    if instruction.type in ['PLACE_ORDER_MARKET', 'PLACE_ORDER_LIMIT'] and instruction.stp == STP.NO_STP:
                        instruction.stp = STP.CANCEL_OLDEST
                    valid_instructions.append(instruction)
                except Exception as ex:
                    bt.logging.warning(f"Error processing instruction submitted by agent {uid} : {ex}\n{instruction}")
            final_instructions = []
            # Enforce the configured limit on maximum submitted instructions (to prevent overloading simulator)
            instructions_per_book = {}
            for instruction in valid_instructions:
                if hasattr(instruction, 'bookId') and instruction.bookId not in instructions_per_book:
                    instructions_per_book[instruction.bookId] = 0
                instructions_per_book[instruction.bookId] += 1
                if instructions_per_book[instruction.bookId] <= self.config.scoring.max_instructions_per_book:
                    final_instructions.append(instruction)
            if len(final_instructions) < len(valid_instructions):
                bt.logging.warning(
                    f"Agent {uid} sent {len(valid_instructions)} instructions "
                    f"(Avg. {len(valid_instructions) / len(instructions_per_book):.2f} / book), "
                    f"with more than {self.config.scoring.max_instructions_per_book} instructions on some books - "
                    f"excess instructions were dropped. Final instruction count {len(final_instructions)}."
                )
                for book_id, count in instructions_per_book.items():
                    bt.logging.trace(f"Agent {uid} Book {book_id} : {count} Instructions")
            # Update the synapse response with only the validated instructions
            synapse.response.instructions = final_instructions
            total_responses += 1
            total_instructions += len(synapse.response.instructions)
        else:
            bt.logging.debug(f"UID {uid} failed to respond : {synapse.dendrite.status_message}")
    return total_responses, total_instructions, success, timeouts, failures

def update_stats(self : Validator, synapses : dict[int, MarketSimulationStateUpdate]) -> None:
    """
    Updates miner request statistics maintained and published by validator

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
        synapses (list[taos.im.protocol.MarketSimulationStateUpdate]): The synapses with attached agent responses to be evaluated for statistics update.
    Returns:
        None
    """
    for uid, synapse in synapses.items():
        self.miner_stats[uid]['requests'] += 1
        if synapse.is_timeout:
            self.miner_stats[uid]['timeouts'] += 1
        elif synapse.is_failure or synapse.response is None:
            self.miner_stats[uid]['failures'] += 1
        elif synapse.is_blacklist:
            self.miner_stats[uid]['rejections'] += 1
        elif synapse.dendrite.process_time:
            self.miner_stats[uid]['call_time'].append(synapse.dendrite.process_time)

async def forward(self: Validator, synapse: MarketSimulationStateUpdate) -> List[FinanceAgentResponse]:
    """
    Forwards state update to miners, validates responses, calculates rewards and handles deregistered UIDs.

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
        synapse : The market state update synapse to be forwarded to miners
    Returns:
        List[FinanceAgentResponse] : Successfully validated responses generated by queried agents.
    """
    responses = []
    if not hasattr(self, '_query_done_event'):
        self._query_done_event = asyncio.Event()
        self._query_done_event.set()
    if self.deregistered_uids != []:
        response = FinanceAgentResponse(agent_id=self.uid)
        response.reset_agents(agent_ids=self.deregistered_uids)
        responses.append(response)

    bt.logging.info(f"Querying Miners...")
    self.querying = True

    session_start = time.time()    
    DendriteManager.configure_session(self)
    bt.logging.debug(f"Session configured ({time.time() - session_start:.4f}s)")

    synapse_start = time.time()
    compress_start = time.time()
    compressed_books = compress(
        synapse.books,
        level=self.config.compression.level,
        engine=self.config.compression.engine,
        version=synapse.version,
    )
    bt.logging.info(f"Compressed books ({time.time()-compress_start:.4f}s).")

    serialized_config = synapse.config.model_dump(mode='json')
    def create_axon_synapse(uid):
        return synapse.model_copy(update={
            "accounts": {uid: synapse.accounts[uid]},
            "notices": {uid: synapse.notices[uid]},
            "config" : serialized_config
        })

    create_start = time.time()
    axon_synapses = {uid: create_axon_synapse(uid) for uid in range(len(self.metagraph.axons))}
    bt.logging.info(f"Created axon synapses ({time.time()-create_start:.4f}s)")

    if self.config.compression.parallel_workers == 0:
        def compress_axon_synapse(synapse):
            return synapse.compress(
                level=self.config.compression.level,
                engine=self.config.compression.engine,
                compressed_books=compressed_books
            )

        axon_synapses = {uid: compress_axon_synapse(axon_synapses[uid]) for uid in range(len(self.metagraph.axons))}
    else:
        num_processes = self.config.compression.parallel_workers if self.config.compression.parallel_workers > 0 else multiprocessing.cpu_count() // 2
        num_axons = len(self.metagraph.axons)
        batches = [self.metagraph.uids[i:i+int(num_axons/num_processes)] for i in range(0, num_axons, int(num_axons/num_processes))]
        axon_synapses = batch_compress(
            axon_synapses,
            compressed_books,
            batches,
            level=self.config.compression.level,
            engine=self.config.compression.engine,
            version=synapse.version
        )
    bt.logging.info(f"Compressed synapses ({time.time()-synapse_start:.4f}s).")

    query_start = time.time()
    synapse_responses = {}
    self._query_done_event.clear()

    async def query_uid(uid):
        try:
            response = await self.dendrite(
                axons=self.metagraph.axons[uid],
                synapse=axon_synapses[uid],
                timeout=self.config.neuron.timeout,
                deserialize=False
            )
            return uid, response
        except asyncio.CancelledError:
            axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                self.metagraph.axons[uid],
                axon_synapses[uid],
                self.config.neuron.timeout
            )
            axon_synapses[uid].dendrite.status_code = 408
            return uid, axon_synapses[uid]
        except Exception as e:
            bt.logging.debug(f"Error querying UID {uid}: {e}")
            axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                self.metagraph.axons[uid],
                axon_synapses[uid],
                self.config.neuron.timeout
            )
            axon_synapses[uid].dendrite.status_code = 500
            return uid, axon_synapses[uid]
    # Create query tasks
    query_tasks = []
    for uid in range(len(self.metagraph.axons)):
        if uid not in self.deregistered_uids:
            query_tasks.append(asyncio.create_task(query_uid(uid)))

    bt.logging.debug(f"Created {len(query_tasks)} query tasks")
    bt.logging.debug(f"Starting query collection")
    global_deadline = time.monotonic() + self.config.neuron.global_query_timeout
    poll_sleep = 0.002
    try:
        while True:
            done_count = sum(1 for t in query_tasks if t.done())
            pending_count = len(query_tasks) - done_count
            now = time.monotonic()
            if pending_count == 0:
                bt.logging.debug(f"Wait completed ({time.time() - query_start:.4f}s | {done_count} done | {pending_count} pending)")
                break
            if now >= global_deadline:
                bt.logging.warning(f"Global timeout hit at {self.config.neuron.global_query_timeout}s; cancelling {pending_count} pending tasks")
                for t in query_tasks:
                    if not t.done():
                        try:
                            t.cancel()
                        except Exception:
                            pass
                break
            await asyncio.sleep(poll_sleep)
    except Exception as e:
        bt.logging.error(f"Wait error at t={time.time() - query_start:.4f}s: {e}")
    collect_deadline = time.time() + 0.05
    collected = 0
    for task in list(query_tasks):
        if time.time() > collect_deadline:
            bt.logging.warning(f"Collection deadline hit, stopping early")
            break
        if task.done():
            try:
                uid, response = task.result()
                synapse_responses[uid] = response
                collected += 1
            except Exception as e:
                bt.logging.debug(f"Task failed during collection: {e}")
    bt.logging.debug(f"Collected {collected} results ({time.time() - query_start:.4f}s)")
    cancelled_count = 0
    for task in query_tasks:
        if not task.done():
            try:
                task.cancel()
                cancelled_count += 1
            except Exception:
                pass

    bt.logging.debug(f"Cancelling {cancelled_count} pending tasks")
    bt.logging.debug(f"Starting cleanup phase at t={time.time() - query_start:.4f}s")

    response_start = time.time()
    missing_count = 0
    for uid in range(len(self.metagraph.axons)):
        if uid not in self.deregistered_uids and uid not in synapse_responses:
            axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                self.metagraph.axons[uid],
                axon_synapses[uid],
                self.config.neuron.timeout
            )
            axon_synapses[uid].dendrite.status_code = 408
            synapse_responses[uid] = axon_synapses[uid]
            missing_count += 1
    bt.logging.debug(f"Created {missing_count} timeout responses in {time.time() - response_start:.4f}s at t={time.time() - query_start:.4f}s")
    query_time = time.time() - query_start
    bt.logging.info(f"Dendrite call completed ({query_time:.4f}s | "
                    f"Timeout {self.config.neuron.timeout}s / {self.config.neuron.global_query_timeout}s).")
    self.dendrite.synapse_history = self.dendrite.synapse_history[-10:]
    
    self.querying = False
    self._query_done_event.set()

    start = time.time()
    total_responses, total_instructions, success, timeouts, failures = validate_responses(self, synapse_responses)
    bt.logging.info(f"Validated Responses ({time.time()-start:.4f}s).")
    start = time.time()
    update_stats(self, synapse_responses)
    bt.logging.info(f"Updated Stats ({time.time()-start:.4f}s).")
    start = time.time()
    responses.extend(set_delays(self, synapse_responses))
    bt.logging.info(f"Set Delays ({time.time()-start:.4f}s).")
    bt.logging.trace(f"Responses: {responses}")
    bt.logging.info(f"Received {total_responses} valid responses containing {total_instructions} instructions "
                    f"({success} SUCCESS | {timeouts} TIMEOUTS | {failures} FAILURES).")
    return responses

async def notify(self : Validator, notices : List[FinanceEventNotification]) -> None:
    """
    Forwards event notifications to the related miner agents.

    Args:
        self (taos.im.neurons.validator.Validator): The intelligent markets simulation validator.
        notices (List[FinanceEventNotification]) : The notice events published by the validator.
    Returns:
        None
    """
    responses = []
    for notice in notices:
        axons = [self.metagraph.axons[notice.event.agentId]] if notice.event.agentId else self.metagraph.axons
        responses.extend(await self.dendrite(
            axons=axons,
            synapse=notice,
            timeout=1
        ))
    for response in responses:
        if response and response.acknowledged:
            bt.logging.info(f"{response[0].type} EventNotification Acknowledged by {response[0].axon.hotkey}")
