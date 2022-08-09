import logging
import threading
import time
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread, Observable):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every `HEARTBEAT_PERIOD` seconds.

    It also maintains a list of active neighbors, which is created by receiving different heartbear messages. 
    Neighbors from which a heartbeat is not received in ``Settings.NODE_TIMEOUT`` will be eliminated

    Communicates with node via observer pattern.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """
    def __init__(self, node_name):
        Observable.__init__(self)
        threading.Thread.__init__(self, name = "heartbeater-" + node_name)
        self.__node_name = node_name
        self.__terminate_flag = threading.Event()

        # List of neighbors
        self.__nodes = {}

    def run(self):
        """
        Send a beat every HEARTBEAT_PERIOD seconds to all the neighbors of the node. 
        Also, it will clear from the neighbors list the nodes that haven't sent a heartbeat in NODE_TIMEOUT seconds. 
        It happend ``HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD`` per HEARTBEAT_PERIOD 
        """
        while not self.__terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.notify(Events.SEND_BEAT_EVENT, None)
            # Wait and refresh node list
            for _ in range(Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD):
                self.clear_nodes()
                time.sleep(Settings.HEARTBEAT_PERIOD/Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD)


    def clear_nodes(self):
        """
        Clear the list of neighbors.
        """
        for n in [node for node,t in list(self.__nodes.items()) if time.time() - t > Settings.NODE_TIMEOUT]:
            logging.debug("({}) Removed {} from the network ".format(self.__node_name, n))
            self.__nodes.pop(n)

    def add_node(self, node):
        """
        Add a node to the list of neighbors.

        Args:
            node (Node): Node to add to the list of neighbors.
        """
        if node != self.__node_name:
            self.__nodes[node] = time.time()

    def get_nodes(self):
        """
        Get the list of actual neighbors.

        Returns:
            list: List of neighbors.
        """
        node_list = list(self.__nodes.keys())
        if self.__node_name not in node_list:
            node_list.append(self.__node_name)
        return node_list

    def stop(self):
        """
        Stop the heartbeater.
        """
        self.__terminate_flag.set()
