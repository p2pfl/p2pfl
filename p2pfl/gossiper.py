import time
import threading

from p2pfl.settings import Settings

class Gossiper(threading.Thread):
    
    # Meterlo como un observer -> Hacer del mismo modo el heartbeater

    def __init__(self, node):
        self.msgs = {}
        self.node = node
        threading.Thread.__init__(self, name=("gossiper-" + node.get_name() ))
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

            messages_left = Settings.GOSSIP_MODEL_SENDS_BY_ROUND
            
            # Lock
            self.add_lock.acquire()
            begin = time.time()
            
            # Send to all the nodes except the ones that the message was already sent to
            if len(self.msgs) > 0:
                msg_list = list(self.msgs.items()).copy()
                nei = set(self.node.get_neighbors())

                for msg,nodes in msg_list:
                    nodes = set(nodes)
                    sended = len(nei - nodes)

                    if messages_left - sended >= 0:
                        self.node.broadcast(msg,exc=list(nodes)) 
                        del self.msgs[msg]
                        if messages_left == 0:
                            break
                    else:
                        # Lists to concatenate / Sets to difference
                        excluded = (list(nei - nodes))[:abs(messages_left - sended)]
                        self.node.broadcast(msg,exc=list(nodes)+excluded)
                        self.msgs[msg] = list(nodes) + list(nei - set(excluded))
                        break

            # Unlock
            self.add_lock.release()
            
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    def stop(self):
        self.terminate_flag.set()