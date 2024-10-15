#
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
"""P2PFL communication tests."""

import time
from typing import Type

import pytest

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.exceptions import (
    CommunicationError,
    NeighborNotConnectedError,
    ProtocolNotStartedError,
)
from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils import set_test_settings, wait_convergence

set_test_settings()


class MockCommand(Command):
    """Mock command for testing purposes."""

    def __init__(self) -> None:
        """Initialize the mock command."""
        self.flag = False

    @staticmethod
    def get_name() -> str:
        """Get the name of the command."""
        return "mock_command"

    def execute(self, *args, **kwargs) -> None:
        """Execute the command."""
        self.flag = True


@pytest.mark.parametrize("protocol_class", [GrpcCommunicationProtocol, InMemoryCommunicationProtocol])
def test_connect_invalid_node(protocol_class):
    """Test that a node can't connect to an invalid node."""
    protocol1 = protocol_class()
    protocol1.start()
    assert protocol1.connect("google.es:80") is False
    assert protocol1.connect("holadani.holaenrique") is False
    assert len(protocol1.get_neighbors()) == 0
    protocol1.stop()


@pytest.mark.parametrize("protocol_class", [GrpcCommunicationProtocol, InMemoryCommunicationProtocol])
def test_basic_communication(protocol_class: Type[CommunicationProtocol]):
    """Test the start and stop methods."""
    # Create 2 communication protocols
    protocol1 = protocol_class()
    protocol2 = protocol_class()

    # Command
    command = MockCommand()
    built_cmd = protocol1.build_msg(command.get_name())

    # Try to send without starting the protocol
    with pytest.raises(ProtocolNotStartedError):
        protocol1.send(protocol2.get_address(), built_cmd, raise_error=True)

    # Try to connect without starting the protocol
    with pytest.raises(ProtocolNotStartedError):
        protocol1.connect(protocol2.get_address())

    # Start the protocols
    protocol1.start()
    protocol2.start()

    # Try to send without connect
    with pytest.raises(NeighborNotConnectedError):
        protocol1.send(protocol2.get_address(), built_cmd, raise_error=True)

    # Connect the protocols
    assert protocol1.connect(protocol2.get_address()) is True

    # Check neighbors
    time.sleep(1)  # Wait for the connection to be established
    assert list(protocol1.get_neighbors().keys()) == [protocol2.get_address()]
    assert list(protocol2.get_neighbors().keys()) == [protocol1.get_address()]

    with pytest.raises(CommunicationError):
        protocol1.send(protocol2.get_address(), built_cmd, raise_error=True, remove_on_error=False)

    # Ensure the command is not executed
    assert command.flag is False

    # Add the command to the first protocol
    protocol2.add_command(command)

    # Send the command
    built_cmd = protocol1.build_msg(command.get_name())  # regenerate the command to refresh the hash
    protocol1.send(protocol2.get_address(), built_cmd, raise_error=True)

    # Ensure the command is executed
    time.sleep(1)  # Wait for the command to be executed
    assert command.flag is True

    # Stop the protocols
    protocol1.stop()
    protocol2.stop()


@pytest.mark.parametrize("protocol_class", [GrpcCommunicationProtocol, InMemoryCommunicationProtocol])
def test_neightboor_management_and_gossip(protocol_class: Type[CommunicationProtocol]):
    """Test the neighbor management."""
    # Create the protocols
    protocol1 = protocol_class()
    protocol2 = protocol_class()
    protocol3 = protocol_class()
    protocol4 = protocol_class()
    protocol5 = protocol_class()

    # Start the protocols
    protocol1.start()
    protocol2.start()
    protocol3.start()
    protocol4.start()
    protocol5.start()

    # Connect the protocols
    protocol1.connect(protocol2.get_address())
    protocol4.connect(protocol5.get_address())
    protocol3.connect(protocol2.get_address())
    protocol3.connect(protocol4.get_address())

    # Wait for convergence
    wait_convergence([protocol1, protocol2, protocol3, protocol4, protocol5], 4, wait=5, only_direct=False)

    # Check neighbors (only direct)
    assert len(protocol1.get_neighbors(only_direct=True)) == 1
    assert len(protocol2.get_neighbors(only_direct=True)) == 2
    assert len(protocol3.get_neighbors(only_direct=True)) == 2
    assert len(protocol4.get_neighbors(only_direct=True)) == 2
    assert len(protocol5.get_neighbors(only_direct=True)) == 1

    # Disconnect 3 from 2
    protocol2.disconnect(protocol3.get_address())

    # Wait for convergence
    wait_convergence([protocol1, protocol2], 1, wait=Settings.HEARTBEAT_TIMEOUT * 2, only_direct=False)
    wait_convergence([protocol3, protocol4, protocol5], 2, wait=Settings.HEARTBEAT_TIMEOUT * 2, only_direct=False)

    # Check neighbors (only direct)
    assert len(protocol1.get_neighbors(only_direct=True)) == 1
    assert len(protocol2.get_neighbors(only_direct=True)) == 1
    assert len(protocol3.get_neighbors(only_direct=True)) == 1
    assert len(protocol4.get_neighbors(only_direct=True)) == 2
    assert len(protocol5.get_neighbors(only_direct=True)) == 1

    # Disconnect 3 from 4
    protocol4.disconnect(protocol3.get_address())

    # Wait for convergence
    wait_convergence([protocol4, protocol5], 1, wait=Settings.HEARTBEAT_TIMEOUT * 2, only_direct=False)
    wait_convergence([protocol1, protocol2], 1, wait=Settings.HEARTBEAT_TIMEOUT * 2, only_direct=False)
    wait_convergence([protocol3], 0, wait=Settings.HEARTBEAT_TIMEOUT * 2, only_direct=False)

    # Check neighbors (only direct)
    assert len(protocol1.get_neighbors(only_direct=True)) == 1
    assert len(protocol2.get_neighbors(only_direct=True)) == 1
    assert len(protocol3.get_neighbors(only_direct=True)) == 0
    assert len(protocol4.get_neighbors(only_direct=True)) == 1
    assert len(protocol5.get_neighbors(only_direct=True)) == 1

    # Stop the protocols
    protocol1.stop()
    protocol2.stop()
    protocol3.stop()
    protocol4.stop()
    protocol5.stop()


@pytest.mark.parametrize("protocol_class", [GrpcCommunicationProtocol, InMemoryCommunicationProtocol])
def test_node_abrupt_down(protocol_class: Type[CommunicationProtocol]):
    """Test that a node abruptly down is removed from the neighbors list."""
    # Create 2 communication protocols
    protocol1 = protocol_class()
    protocol2 = protocol_class()

    # Start the protocols
    protocol1.start()
    protocol2.start()

    # Connect the protocols
    protocol1.connect(protocol2.get_address())

    # Wait for convergence
    wait_convergence([protocol1, protocol2], 1, wait=5, only_direct=True)

    # Check neighbors
    assert len(protocol1.get_neighbors()) == 1
    assert len(protocol2.get_neighbors()) == 1

    # Stop the protocol 2
    protocol2.stop()

    # Wait for convergence
    wait_convergence([protocol1], 0, wait=5, only_direct=True)

    # Check neighbors
    assert len(protocol1.get_neighbors()) == 0

    # Stop the protocol 1
    protocol1.stop()
