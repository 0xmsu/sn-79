# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import argparse
import bittensor as bt
from typing import Union
from prometheus_client import start_http_server
from enum import Enum

class prometheus:
    """Namespace for prometheus tooling."""

    # Prometheus global logging levels.
    class level(Enum):
        OFF = "OFF"
        INFO = "INFO"
        DEBUG = "DEBUG"

        def __str__(self):
            return self.value

    # Prometheus Global state.
    port: int = None
    started: bool = False

    def __new__(
        cls,
        config: "bt.Config" = None,
        port: int = None,
        level: Union[str, "prometheus.level"] = None
    ):
        """
        Instantiates a global prometheus DB which can be accessed by other processes.
        Each prometheus DB is designated by a port.
        Args:
            config (:obj:`bt.Config`, `optional`, defaults to bt.prometheus.config()):
                A config namespace object created by calling bt.prometheus.config()
            port (:obj:`int`, `optional`, defaults to bt.defaults.prometheus.port ):
                The port to run the prometheus DB on, this uniquely identifies the prometheus DB.
            level (:obj:`prometheus.level`, `optional`, defaults to bt.defaults.prometheus.level ):
                Prometheus logging level. If OFF, the prometheus DB is not initialized.
        """
        if config == None:
            config = prometheus.config()

        if isinstance(level, prometheus.level):
            level = level.name  # Convert ENUM to str.

        config.prometheus.port = port if port != None else config.prometheus.port
        config.prometheus.level = level if level != None else config.prometheus.level

        if isinstance(config.prometheus.level, str):
            config.prometheus.level = (
                config.prometheus.level.upper()
            )  # Convert str to upper case.

        cls.check_config(config)

        return cls.serve(
            cls,
            port=config.prometheus.port,
            level=config.prometheus.level,
        )

    def serve(cls, port, level) -> bool:
        """
        Starts the prometheus client server on the configured port.
        Args:
            port (:obj:`int`, `optional`, defaults to bt.defaults.prometheus.port ):
                The port to run the prometheus DB on, this uniquely identifies the prometheus DB.
            level (:obj:`prometheus.level`, `optional`, defaults to bt.defaults.prometheus.level ):
                Prometheus logging level. If OFF, the prometheus DB is not initialized.
        """
        if level == prometheus.level.OFF.name:  # If prometheus is off, return true.
            bt.logging.success("Prometheus:".ljust(20) + "<red>OFF</red>")
            return True
        else:
            try:
                start_http_server(port)
            except OSError:
                # The singleton process is likely already running.
                bt.logging.error(
                    f"Prometheus: Port {port} Already in use!"
                )
            prometheus.started = True
            prometheus.port = port
            bt.logging.success(
                f"Prometheus: ON using: [::]:{port}"
            )
            return True

    @classmethod
    def config(cls) -> "bt.Config":
        """
        Get config from the argument parser
        
        Return: bt.Config object
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser=parser)
        return bt.Config(parser)

    @classmethod
    def help(cls):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser"""
        try:
            parser.add_argument(
                "--prometheus.port",
                type=int,
                required=False,
                default=9001,
                help="""Prometheus serving port.""",
            )
            parser.add_argument(
                "--prometheus.level",
                required=False,
                type=str,
                choices=[l.name for l in list(prometheus.level)],
                default="INFO",
                help="""Prometheus logging level. <OFF | INFO | DEBUG>""",
            )
        except argparse.ArgumentError as e:
            pass

    @classmethod
    def check_config(cls, config: "bt.Config"):
        """Check config for wallet name/hotkey/path/hotkeys/sort_by"""
        assert "prometheus" in config
        assert config.prometheus.level in [
            l.name for l in list(prometheus.level)
        ], "config.prometheus.level must be in: {}".format(
            [l.name for l in list(prometheus.level)]
        )
        assert (
            config.prometheus.port > 1024 and config.prometheus.port < 65535
        ), "config.prometheus.port must be in range [1024, 65535]"
        if "axon" in config and "port" in config.axon:
            assert (
                config.prometheus.port != config.axon.port
            ), "config.prometheus.port != config.axon.port"