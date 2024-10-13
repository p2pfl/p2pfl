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

"""Edge node client."""

import asyncio

import grpc

from p2pfl.communication.protocols.edge.proto.edge_node_pb2 import EdgeMessage
from p2pfl.communication.protocols.edge.proto.edge_node_pb2_grpc import NodeStub
from p2pfl.management.logger import logger

class EdgeClientConnection:
    def __init__(self, addr, callbacks):
        self.addr = addr
        self.pending_messages = asyncio.Queue(maxsize=10)
        self.started = False
        self.callbacks = callbacks

    async def start(self):
        try:
            # Check if already started
            if self.started:
                return
            self.started = True
            # Create connection
            channel = grpc.aio.insecure_channel(self.addr)
            stub = NodeStub(channel)
            # Create a stream (async generators -> in/out)
            print(f"Client connected to {self.addr}")
            await asyncio.create_task(
                self.__process_messages(
                    stub.MainStream(
                        self.__get_pending_messages()
                    )
                )
            )
        except grpc.aio.AioRpcError as e:
            logger.error("EdgeNode", f"Something went wrong with the EdgeClientConnection: {e.details()}")
        finally:
            await channel.close()
            self.started = False

    ###
    # Process/Generate Messages
    ###

    async def __get_pending_messages(self):
        while True:
            yield await self.pending_messages.get()


    async def __process_messages(self, responses):
        async for response in responses:
            asyncio.create_task(self.__run_callback(response))

    async def __run_callback(self, response):
        # Get and run callback
        work = asyncio.to_thread(self.callbacks[response.cmd], response.message, response.weights)
        result_msg, result_weights = await asyncio.create_task(work)

        print(f"Result: {result_msg}")
        print(f"Weights: {result_weights is not None}")

        # Response
        await self.pending_messages.put(
            EdgeMessage(
                id=response.id,
                cmd=response.cmd,
                message=result_msg,
                weights=result_weights
            )
        )
