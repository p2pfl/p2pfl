
# PATRÓn ESTRATEGIA

import threading
import logging

class FedAvg(threading.Thread):
    def __init__(self, n):
        threading.Thread.__init__(self)
        self.node = n
        self.models = []
        self.lock = threading.Lock()

    def add_model(self, m):
        self.lock.acquire()

        self.models.append(m)
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)) + ")")

        #Revisamos si están todos
        if len(self.models)==len(self.node.neightboors):
            self.start()
        
        self.lock.release()
        
    def clear_models(self):
        self.models = []

    def run(self):
        sum=self.node.model
        for m in self.models:
            sum=sum+m
        self.node.model=sum/(len(self.models)+1)
        self.node.round = self.node.round + 1
        logging.info("Promediación realizada: " + str(self.node.model))
        self.clear_models()
        