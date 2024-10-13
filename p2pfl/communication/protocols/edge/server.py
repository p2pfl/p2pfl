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

"""Edge node server."""

import asyncio
from concurrent import futures
import threading
from typing import Dict, Tuple

import grpc

from p2pfl.communication.protocols.edge.proto.edge_node_pb2 import EdgeMessage
from p2pfl.communication.protocols.edge.proto.edge_node_pb2_grpc import NodeServicer, add_NodeServicer_to_server


class EdgeServerConnection(NodeServicer):
    def __init__(self, addr):
        self.addr = addr
        self.active_connections: Dict[str, Tuple[int, asyncio.Queue]] = {}
        self.client_responses: Dict[str, Dict[int, asyncio.Future]] = {} #ESTO ESTA MALLLL DEBE SER POR CADA CONECION
        self.loop = None

    ###
    # Async/Blocking start/stop Methods
    ###

    async def async_start(self):
        self.server = grpc.aio.server()
        add_NodeServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.addr)
        await self.server.start()
        await self.server.wait_for_termination() # Reemplazar por uno custom
        await self.server.stop(0)

    def start(self):
        # Create loop thead
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        self.loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_event_loop, args=(self.loop,), daemon=True)
        thread.start()

        # Start server
        asyncio.run_coroutine_threadsafe(self.async_start(), self.loop)

    async def stop(self):
        raise NotImplementedError

    ###
    # Stream Init
    ###

    async def MainStream(self, request_iterator, context):
        # Register the connection
        queue = asyncio.Queue(maxsize=10)
        self.active_connections[context.peer()] = (0, queue)
        print(f"Client connected from {context.peer()}")

        # Process incoming messages
        asyncio.create_task(self.__process_messages(request_iterator, context.peer()))

        # Send pending messages
        while True:
            yield await queue.get()

    ###
    # Process/Generate Messages
    ###

    async def __process_messages(self, request_iterator, peer):
        self.client_responses[peer] = {}
        async for request in request_iterator:
            # Check if it's a response
            if request.id in self.client_responses[peer]:
                self.client_responses[peer][request.id].set_result(request) 
        # Remove connection
        del self.active_connections[peer]

    async def async_send_message(self, address, cmd, message=None, weights=None, timeout=60):

        # Get and increment message ID
        message_id = self.active_connections[address][0]
        self.active_connections[address] = (message_id + 1, self.active_connections[address][1])

        # Create a Future to await the response
        self.client_responses[address][message_id] = asyncio.Future()

        # Send message
        await self.active_connections[address][1].put(EdgeMessage(id=message_id, cmd=cmd, message=message, weights=weights))

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(self.client_responses[address][message_id], timeout)
        except asyncio.TimeoutError:
            print(f"Timeout waiting for response from {address} for message ID {message_id}")
            del self.client_responses[address][message_id]  # Clean up after timeout
            return None  # or raise an exception, depending on your needs

        del self.client_responses[address][message_id]  # Clean up after receiving
        return response


    async def async_broadcast_message(self, cmd, message=None, weights=None, timeout=60):
        """Broadcasts a message to all connected clients."""
        # Send message
        future_msgs = [asyncio.create_task(self.async_send_message(address, cmd, message, weights, timeout=timeout)) for address in self.active_connections]
        # Wait for responses
        return {address: await future for address, future in zip(self.active_connections, future_msgs)}

    def broadcast_message(self, cmd, message=None, weights=None, timeout=60):
        return asyncio.run_coroutine_threadsafe(self.async_broadcast_message(cmd, message, weights, timeout), self.loop).result()