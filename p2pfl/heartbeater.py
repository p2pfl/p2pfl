from platform import node
import threading
import time
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread, Observable):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every `HEARTBEAT_PERIOD` seconds.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """
    def __init__(self, node_name):
        Observable.__init__(self)
        threading.Thread.__init__(self, name = "heartbeater-" + node_name)
        self.terminate_flag = threading.Event()

        # List of neighbors
        self.nodes = {}

    def run(self):
        """
        Send a beat every HEARTBEAT_PERIOD seconds to all the neighbors of the node.
        """
        while not self.terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.notify(Events.SEND_BEAT_EVENT, None)
            # Wait and refresh node list
            for _ in range(Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD):
                self.clear_nodes()
                time.sleep(Settings.HEARTBEAT_PERIOD/Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD)

    def stop(self):
        self.terminate_flag.set()

    def clear_nodes(self):
        """
        Clear the list of neighbors.
        """
        for n in [node for node,t in list(self.nodes.items()) if time.time() - t > Settings.NODE_TIMEOUT]:
            self.nodes.pop(n)
            self.notify(Events.NODE_DISCONNECTED, None)

    def add_node(self, node):
        """
        Add a node to the list of neighbors.

        Args:
            node (Node): Node to add to the list of neighbors.
        """
        self.nodes[node] = time.time()

    def get_nodes(self):
        """
        Get the list of actual neighbors.

        Returns:
            list: List of neighbors.
        """
        return list(self.nodes.keys())

