import time
import threading
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

##################
#    Gossiper    #
##################

class Gossiper(threading.Thread, Observable):
    """
    Thread based gossiper. It gossip messages from list of pending messages. ``Settings.GOSSIP_MESSAGES_PER_ROUND`` are sended per iteration (`Settings.GOSSIP_FREC` times per second).

    Communicates with node via observer pattern.
    
    Args:
        nodo_padre (str): Name of the parent node.
        neighbors (list): List of neighbors.

    """
    def __init__(self, node_name, neighbors):
        Observable.__init__(self)
        threading.Thread.__init__(self, name=("gossiper-" + node_name ))
        self.__neighbors = neighbors # list as reference of the original neighbors list
        self.__msgs = {}
        self.__add_lock = threading.Lock()
        self.__terminate_flag = threading.Event()

    def add_messages(self, msgs, node):
        """
        Add messages to the list of pending messages.

        Args:
            msgs (list): List of messages to add.
            node (Node): Node that sent the messages.
        """
        self.__add_lock.acquire()
        for msg in msgs:
            self.__msgs[msg] = [node]
        self.__add_lock.release()

    def run(self):
        """
        Gossiper Main Loop. Sends `Settings.GOSSIP_MODEL_SENDS_BY_ROUND` messages `Settings.GOSSIP_FREC` times per second.        
        """
        while not self.__terminate_flag.is_set():

            messages_left = Settings.GOSSIP_MESSAGES_PER_ROUND
            
            # Lock
            self.__add_lock.acquire()
            begin = time.time()
            
            # Send to all the nodes except the ones that the message was already sent to
            if len(self.__msgs) > 0:
                msg_list = list(self.__msgs.items()).copy()
                nei = set(self.__neighbors.copy()) # copy to avoid concurrent problems

                for msg,nodes in msg_list:
                    nodes = set(nodes)
                    sended = len(nei - nodes)

                    if messages_left - sended >= 0:
                        self.notify(Events.GOSSIP_BROADCAST_EVENT, (msg,list(nodes)))
                        del self.__msgs[msg]
                        messages_left = messages_left - sended
                        if messages_left == 0:
                            break
                    else:
                        # Lists to concatenate / Sets to difference
                        excluded = (list(nei - nodes))[:abs(messages_left - sended)]
                        self.notify(Events.GOSSIP_BROADCAST_EVENT, (msg,list(nodes)+excluded))
                        self.__msgs[msg] = list(nodes) + list(nei - set(excluded))
                        break

            # Unlock
            self.__add_lock.release()
            
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_MESSAGES_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    def stop(self):
        """
        Stop the gossiper.
        """
        self.__terminate_flag.set()