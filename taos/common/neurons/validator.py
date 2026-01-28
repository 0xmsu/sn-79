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

import copy
import torch
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List
from traceback import print_exception

from abc import abstractmethod

from taos.common.neurons import BaseNeuron
from taos.mock import MockDendrite
from taos.common.config import add_validator_args

import taos.common.utils.weights as weight_utils


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.deregistered_uids = []

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.Dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        self.scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )        
        self.pending_weights = []
        self.last_commit = None

        # Init sync with the network. Updates the metagraph.
        self.sync(save_state=False)

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("`neuron.axon_off=True` - IP will not be served to chain.")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        bt.logging.info(f"Validator Started on net{self.config.netuid}! Address : {self.dendrite.keypair.ss58_address} | Stake : {self.metagraph.stake[self.uid]}τ | Alpha Stake : {self.subtensor.get_stake_for_hotkey(self.dendrite.keypair.ss58_address, self.config.netuid)}")

    def serve_axon(self):
        """Serve axon to enable external connections and advertise the IP of the validator to allow scraping of metrics."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    # The `run` function is not used by this subnet since the validator is launched as a FastAPI client in order to receive communications from the simulator.
    def run(self):
        pass

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def prepare_weights(self):        
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. 
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        
        If burn_uid and burn_ratio are configured, automatically allocates a percentage of emissions to the burn address.
        """
        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in the reward function."
            )

        bt.logging.debug(f"Processing Scores: {self.scores}")
        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        weight_scores = self.scores
        if min(self.scores) < 0:
            weight_scores = self.scores - min(self.scores)
        raw_weights = torch.nn.functional.normalize(weight_scores, p=1, dim=0)

        # Apply burn mechanism if configured
        burn_uid = getattr(self.config.neuron, 'burn_uid', None)
        burn_ratio = getattr(self.config.neuron, 'burn_ratio', 0.0)
        
        if burn_uid is not None and burn_ratio > 0:
            # Validate burn_ratio is between 0 and 1
            burn_ratio = max(0.0, min(1.0, burn_ratio))
            
            # Validate burn_uid exists in metagraph
            if burn_uid >= len(raw_weights):
                bt.logging.warning(
                    f"Burn UID {burn_uid} is out of range (metagraph size: {len(raw_weights)}). Skipping burn allocation."
                )
            else:
                bt.logging.info(f"Applying {burn_ratio*100:.2f}% emission burn to UID {burn_uid}")
                
                # Scale down all existing weights by (1 - burn_ratio)
                raw_weights = raw_weights * (1.0 - burn_ratio)
                
                # Allocate burn_ratio to the burn_uid
                raw_weights[burn_uid] = burn_ratio
                
                # Renormalize to ensure sum is 1.0
                raw_weights = torch.nn.functional.normalize(raw_weights, p=1, dim=0)
                
                bt.logging.debug(f"Post-burn weight for UID {burn_uid}: {raw_weights[burn_uid]:.6f}")

        bt.logging.debug(f"raw_weights={raw_weights}")
        bt.logging.debug(f"raw_weight_uids={self.metagraph.uids}")
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights.to("cpu").numpy(),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug(f"processed_weights={processed_weights}")
        bt.logging.debug(f"processed_weight_uids={processed_weight_uids}")

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug(f"uint_weights={uint_weights}")
        bt.logging.debug(f"uint_uids={uint_uids}")
        return uint_uids, uint_weights

    def set_weights(self):
        """
        Weight setting function
        """
        # Get relevant hyperparameters
        version_key = self.hyperparams.weights_version
        commit_reveal_weights_enabled = bool(self.hyperparams.commit_reveal_weights_enabled)
        # Prepare weights for submission
        uint_uids, uint_weights = self.prepare_weights()
        bt.logging.info(f"`commit_reveal_weights_enabled` : {commit_reveal_weights_enabled}")
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            version_key=self.spec_version,
        )
        return result

    @abstractmethod
    def handle_deregistration(self, uid):
        """
        Abstract method to enable specific handling of hotkey deregistration by the validator
        """
        ...

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.trace("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.       
        bt.logging.debug("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons and len(self.hotkeys) == len(self.metagraph.hotkeys):            
            bt.logging.debug("No axon changes!")
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.handle_deregistration(uid)

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            bt.logging.debug("Handling new hotkeys...")
            new_moving_average = torch.zeros((self.metagraph.n)).to(
                self.device
            )
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        bt.logging.debug("Updating Scores...")
        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        bt.logging.debug("Cloning UIDs...")
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach().to(self.device)
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        bt.logging.debug("Scattering...")
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, rewards
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {scattered_rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        bt.logging.debug("Calculating MA...")
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.trace("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = torch.load(self.config.neuron.full_path + "/state.pt", weights_only=False)
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
