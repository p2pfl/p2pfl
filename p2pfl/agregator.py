
# PATRÓn ESTRATEGIA

# Patron obervador -> notificar cuando acabe agregación

import threading
import logging


#dejarlo mas pro referenciando el paper y así

# acordarse de ponderar los pesos

#en algún lado se tendrá que validar el modelo recibido, cuando llegue aquí debe estar BIEN VALIDADO si o si

class FedAvg(threading.Thread):
    def __init__(self, n):
        threading.Thread.__init__(self)
        self.node = n
        self.models = []
        self.lock = threading.Lock()


    def add_model(self, m):
        self.lock.acquire()

        self.models.append(m)
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)+1) + ")")



        #Revisamos si están todos
        if len(self.models)==(len(self.node.neightboors)+1):
            self.start()
        
        self.lock.release()
        
    def clear(self):
        self.__init__(self.node)

    def run(self):
        logging.info("Agregating models.")
        self.node.learner.set_parameters(FedAvg.agregate(self.models))
        self.clear()
        # Notificamos al nodo
        self.node.on_round_finished()

    def agregate(models):
        # (MEAN)
        # Sum
        sum= models[-1]
        for m in models[:-1]:

            for layer in m:
                sum[layer] = sum[layer] + m[layer]

        
        
        # Dividimos por el número de modelos
        for layer in sum:
            sum[layer] = sum[layer]/(len(models))

        return sum
            
        