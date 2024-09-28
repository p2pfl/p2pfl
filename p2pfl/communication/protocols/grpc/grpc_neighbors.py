#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""gRPC neighbors."""

import time
from os.path import isfile
from typing import Optional, Tuple

import grpc
from p2pfl.communication.protocols.grpc.proto import node_pb2, node_pb2_grpc
from p2pfl.communication.protocols.neighbors import Neighbors
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class GrpcNeighbors(Neighbors):
    """Implementation of the neighbors for a GRPC communication protocol."""

    def refresh_or_add(self, addr: str, time: float) -> None:
        """
        Refresh or add a neighbor.

        Args:
            addr: Address of the neighbor.
            time: Time of the last heartbeat.

        """
        # Update if exists
        if addr in self.neis:
            # Update time
            self.neis_lock.acquire()
            self.neis[addr] = (
                self.neis[addr][0],
                self.neis[addr][1],
                time,
            )
            self.neis_lock.release()
        else:
            # Add
            self.add(addr, non_direct=True)

    def connect(
        self, addr: str, non_direct: bool = False, handshake_msg: bool = True
    ) -> Tuple[Optional[grpc.Channel], Optional[node_pb2_grpc.NodeServicesStub], float]:
        """
        Connect to a neighbor.

        Args:
            addr: Address of the neighbor to connect.
            non_direct: If the connection is direct or not.
            handshake_msg: If a handshake message is needed.

        """
        if non_direct:
            logger.debug(self.self_addr, f"🔍 Found node {addr}")
            return self.__build_non_direct_neighbor(addr)
        else:
            logger.info(self.self_addr, f"🤝 Adding {addr}")
            return self.__build_direct_neighbor(addr, handshake_msg)

    def __build_direct_neighbor(
        self, addr: str, handshake_msg: bool
    ) -> Tuple[Optional[grpc.Channel], Optional[node_pb2_grpc.NodeServicesStub], float]:
        """
        Build a direct neighbor.

        .. todo:: Remove this, some code duplication and in real enviroments this wont be allowed.
        """
        try:
            # Create channel and stub
            if Settings.USE_SSL and isfile(Settings.SERVER_CRT):
                with open(Settings.CLIENT_KEY) as key_file, open(Settings.CLIENT_CRT) as crt_file, open(Settings.CA_CRT) as ca_file:
                    private_key = key_file.read().encode()
                    certificate_chain = crt_file.read().encode()
                    root_certificates = ca_file.read().encode()
                creds = grpc.ssl_channel_credentials(
                    root_certificates=root_certificates, private_key=private_key, certificate_chain=certificate_chain
                )
                channel = grpc.secure_channel(addr, creds)
            else:
                channel = grpc.insecure_channel(addr)
            stub = node_pb2_grpc.NodeServicesStub(channel)

            if not stub:
                raise Exception(f"Cannot create a stub for {addr}")

            # Handshake
            if handshake_msg:
                res = stub.handshake(
                    node_pb2.HandShakeRequest(addr=self.self_addr),
                    timeout=Settings.GRPC_TIMEOUT,
                )
                if res.error:
                    logger.info(self.self_addr, f"Cannot add a neighbor: {res.error}")
                    channel.close()
                    raise Exception(f"Cannot add a neighbor: {res.error}")

            # Add neighbor
            return (channel, stub, time.time())

        except Exception as e:
            logger.info(self.self_addr, f"Crash while adding a neighbor: {e}")
            # Re-raise exception
            raise e

    def __build_non_direct_neighbor(self, _: str) -> Tuple[None, None, float]:
        return (None, None, time.time())

    def disconnect(self, addr: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            addr: Address of the neighbor to disconnect.
            disconnect_msg: If a disconnect message is needed.

        """
        try:
            # If the other node still connected, disconnect
            node_channel, node_stub, _ = self.get(addr)
            if disconnect_msg:
                if node_stub is not None:
                    node_stub.disconnect(node_pb2.HandShakeRequest(addr=self.self_addr))
                # Close channel
                if node_channel is not None:
                    node_channel.close()
        except Exception:
            pass
