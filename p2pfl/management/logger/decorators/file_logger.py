#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""File Logger Decorator."""

import logging
import os
from logging.handlers import RotatingFileHandler

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.settings import Settings


class FileLogger(LoggerDecorator):
    """File logger decorator."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the logger."""
        super().__init__(p2pflogger)

        # Setup the file handler for logging
        self.setup_file_handler()

    def setup_file_handler(self) -> None:
        """Set up the file handler for logging."""
        if not os.path.exists(Settings.general.LOG_DIR):
            os.makedirs(Settings.general.LOG_DIR)

        # Find existing run log files
        existing_runs = [
            f
            for f in os.listdir(Settings.general.LOG_DIR)
            if os.path.isfile(os.path.join(Settings.general.LOG_DIR, f)) and f.startswith("run-") and f.endswith(".log")
        ]

        # Extract run numbers and their corresponding files
        run_files = []
        for run_file in existing_runs:
            try:
                run_num = int(run_file.split("-")[1].split(".")[0])
                run_files.append((run_num, run_file))
            except (IndexError, ValueError):
                continue

        # Sort by run number (oldest first)
        run_files.sort()

        # Delete oldest files if we exceed the maximum
        max_runs = Settings.general.MAX_LOG_RUNS
        if len(run_files) >= max_runs:
            # Keep the most recent (max_runs - 1) files to make room for the new one
            files_to_delete = run_files[: -(max_runs - 1)] if max_runs > 1 else run_files
            for _, file_to_delete in files_to_delete:
                try:
                    os.remove(os.path.join(Settings.general.LOG_DIR, file_to_delete))
                except OSError:
                    # Log but continue if we can't delete a file
                    print(f"Warning: Could not delete old log file {file_to_delete}")

        # Determine the next run ID
        run_id = 1
        if run_files:
            run_id = run_files[-1][0] + 1 if run_files else 1

        log_filename = f"{Settings.general.LOG_DIR}/run-{run_id}.log"

        file_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=3)
        file_formatter = logging.Formatter(
            "[ %(asctime)s | %(node)s | %(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.add_handler(file_handler)
