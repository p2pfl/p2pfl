import grpc
import time
from typing import Optional, Tuple
from p2pfl.communication.grpc.proto import node_pb2, node_pb2_grpc
from p2pfl.communication.neightbors import Neighbors
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class GrpcNeighbors(Neighbors):

    def refresh_or_add(self, addr: str, time: time) -> None:
        # Update if exists
        if addr in self.neis.keys():
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
    ) -> Tuple[Optional[grpc.Channel], Optional[grpc.Channel], float]:
        if non_direct:
            return self.__build_non_direct_neighbor(addr)
        else:
            return self.__build_direct_neighbor(addr, handshake_msg)

    def __build_direct_neighbor(self, addr: str, handshake_msg: bool) -> bool:
        try:
            # Create channel and stub
            channel = grpc.insecure_channel(addr)
            stub = node_pb2_grpc.NodeServicesStub(channel)

            # Handshake
            if handshake_msg:
                res = stub.handshake(
                    node_pb2.HandShakeRequest(addr=self.self_addr),
                    timeout=Settings.GRPC_TIMEOUT,
                )
                if res.error:
                    logger.info(self.self_addr, f"Cannot add a neighbor: {res.error}")
                    channel.close()
                    return False

            # Add neighbor
            return (channel, stub, time.time())

        except Exception as e:
            logger.info(self.self_addr, f"Crash while adding a neighbor: {e}")
            # Re-raise exception
            raise e

    def __build_non_direct_neighbor(self, _: str) -> bool:
        return (None, None, time.time())

    def disconnect(self, addr: str, disconnect_msg: bool = True) -> None:
        try:
            # If the other node still connected, disconnect
            node_channel, node_stub, _ = self.get(addr)
            if disconnect_msg:
                if node_stub is not None:
                    node_stub.disconnect(
                        node_pb2.HandShakeRequest(addr=self.self_addr)
                    )
                # Close channel
                if node_channel is not None:
                    node_channel.close()
        except Exception:
            pass
