
# PATRÓn ESTRATEGIA

import threading

class FedAvg(threading.Thread):
    def __init__(self, n):
        self.n = n

    def run(self):
        for i in range(self.n):
            print(i)