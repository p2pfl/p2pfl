# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Message storage for logging communication events."""

import datetime
from threading import Lock
from typing import Any

# Define the storage structure types
MessageEntryType = dict[str, Any]  # Individual message entry with details
NodeMessagesType = dict[str, list[MessageEntryType]]  # Messages by node
DirectionMessagesType = dict[str, NodeMessagesType]  # Messages by direction (sent/received)
MessageResultType = list[MessageEntryType]  # Return type for filtered messages


class MessageStorage:
    """
    Message storage. Stores communication events between nodes.

    Format:

    .. code-block:: python
        [
            {
                "timestamp": datetime.datetime,
                "source": "source_node",
                "destination": "dest_node",
                "direction": "sent"/"received",
                "cmd": "command_name",
                "package_type": "message"/"weights",
                "package_size": size_in_bytes,
                "round": round_number,
                "additional_info": {...} or None
            },
        ]
    """

    def __init__(self, disable_locks: bool = False) -> None:
        """Initialize the message storage."""
        self.messages: list[MessageEntryType] = []
        self.lock = Lock() if not disable_locks else None

    def add_message(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message entry to the storage.

        Args:
            node: The node address.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes (if available).
            round_num: The federated learning round number (if applicable).
            additional_info: Additional information as a dictionary.

        """
        # Lock
        if self.lock:
            self.lock.acquire()

        try:
            # Ensure direction is valid
            if direction not in ["sent", "received"]:
                raise ValueError(f"Invalid direction: {direction}. Must be 'sent' or 'received'.")

            # Create the message entry
            message_entry = {
                "timestamp": datetime.datetime.now(),
                "cmd": cmd,
                "direction": direction,
                "package_type": package_type,
                "package_size": package_size,
                "round": 0 if round_num is None else round_num,
            }

            # Add source and destination based on direction
            if direction == "sent":
                message_entry["source"] = node
                message_entry["destination"] = source_dest
            else:
                message_entry["source"] = source_dest
                message_entry["destination"] = node

            # Add additional info if provided
            if additional_info:
                message_entry["additional_info"] = additional_info
            else:
                message_entry["additional_info"] = None

            # Add to storage
            self.messages.append(message_entry)
        finally:
            # Unlock
            if self.lock:
                self.lock.release()

    def get_messages(
        self,
        node: str | None = None,
        direction: str | None = None,
        cmd: str | None = None,
        round_num: int | None = None,
        limit: int | None = None,
    ) -> list[MessageEntryType]:
        """
        Get messages with optional filtering.

        Args:
            node: Filter by node address (as source or destination) (optional).
            direction: Filter by direction ("sent" or "received") (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned (optional).

        Returns:
            A list of message dictionaries matching the filters.

        """
        # Lock
        if self.lock:
            self.lock.acquire()

        try:
            # Start with a copy of all messages
            filtered_messages = self.messages.copy()

            # Filter by direction if provided
            if direction:
                if direction not in ["sent", "received"]:
                    raise ValueError(f"Invalid direction: {direction}. Must be 'sent' or 'received'.")
                filtered_messages = [msg for msg in filtered_messages if msg["direction"] == direction]

            # Filter by node if provided (node could be source or destination based on direction)
            if node:
                if direction == "sent":
                    filtered_messages = [msg for msg in filtered_messages if msg["source"] == node]
                elif direction == "received":
                    filtered_messages = [msg for msg in filtered_messages if msg["destination"] == node]
                else:
                    # If no direction specified, match either source or destination
                    filtered_messages = [msg for msg in filtered_messages if msg["source"] == node or msg["destination"] == node]

            # Filter by command if provided
            if cmd:
                filtered_messages = [msg for msg in filtered_messages if msg["cmd"] == cmd]

            # Filter by round if provided
            if round_num is not None:
                filtered_messages = [msg for msg in filtered_messages if msg.get("round") == round_num]

            # Apply limit if provided
            if limit is not None and filtered_messages:
                filtered_messages = filtered_messages[-limit:]

            return filtered_messages
        finally:
            # Unlock
            if self.lock:
                self.lock.release()

    def get_sent_messages(
        self, node: str | None = None, cmd: str | None = None, round_num: int | None = None, limit: int | None = None
    ) -> list[MessageEntryType]:
        """
        Get sent messages with optional filtering.

        Args:
            node: Filter by source node address (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned (optional).

        Returns:
            Sent messages matching the filters.

        """
        return self.get_messages(node=node, direction="sent", cmd=cmd, round_num=round_num, limit=limit)

    def get_received_messages(
        self, node: str | None = None, cmd: str | None = None, round_num: int | None = None, limit: int | None = None
    ) -> list[MessageEntryType]:
        """
        Get received messages with optional filtering.

        Args:
            node: Filter by destination node address (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned (optional).

        Returns:
            Received messages matching the filters.

        """
        return self.get_messages(node=node, direction="received", cmd=cmd, round_num=round_num, limit=limit)
