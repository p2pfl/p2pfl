import time
import threading
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

class Gossiper(threading.Thread, Observable):
    
    # Meterlo como un observer -> Hacer del mismo modo el heartbeater

    def __init__(self, node_name, neighbors):
        Observable.__init__(self)
        threading.Thread.__init__(self, name=("gossiper-" + node_name ))
        self.neighbors = neighbors # list as reference of the original neighbors list
        self.msgs = {}
        self.add_lock = threading.Lock()
        self.terminate_flag = threading.Event()

    def add_messages(self, msgs, node):
        self.add_lock.acquire()
        for msg in msgs:
            self.msgs[msg] = [node]
        self.add_lock.release()

    def run(self):
        """
        Gossiper Main Loop.
        Sends `Settings.GOSSIP_MODEL_SENDS_BY_ROUND` messages `Settings.GOSSIP_FREC` times per second.        
        """
        while not self.terminate_flag.is_set():

            messages_left = Settings.GOSSIP_MESSAGES_PER_ROUND
            
            # Lock
            self.add_lock.acquire()
            begin = time.time()
            
            # Send to all the nodes except the ones that the message was already sent to
            if len(self.msgs) > 0:
                msg_list = list(self.msgs.items()).copy()
                nei = set(self.neighbors)

                for msg,nodes in msg_list:
                    nodes = set(nodes)
                    sended = len(nei - nodes)

                    if messages_left - sended >= 0:
                        self.notify(Events.GOSSIP_BROADCAST_EVENT, (msg,list(nodes)))
                        del self.msgs[msg]
                        if messages_left == 0:
                            break
                    else:
                        # Lists to concatenate / Sets to difference
                        excluded = (list(nei - nodes))[:abs(messages_left - sended)]
                        self.notify(Events.GOSSIP_BROADCAST_EVENT, (msg,list(nodes)+excluded))
                        self.msgs[msg] = list(nodes) + list(nei - set(excluded))
                        break

            # Unlock
            self.add_lock.release()
            
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_MESSAGES_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    def stop(self):
        self.terminate_flag.set()