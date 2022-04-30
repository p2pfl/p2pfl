
# PATRÓn ESTRATEGIA

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
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)) + ")")

        #Revisamos si están todos
        if len(self.models)==len(self.node.neightboors):
            print("Se debería promediar pero no lo hacemos aún.")
            #self.start()
        
        self.lock.release()
        
    def clear_models(self):
        self.models = []

    def run(self):
        self.models.append(self.node.model) # agregamos el modelo del propio nodo
        self.node.model = FedAvg.agregate(self.models)
        self.node.round = self.node.round + 1
        logging.info("Promediación realizada: ")
        self.clear_models()

    def agregate(models):
        # (MEAN)
        # Sum
        sum=models[-1]
        for m in models[:-1]:
            for layer in m:
                sum[layer]+=m[layer]

        # Dividimos por el número de modelos
        for layer in sum:
            sum[layer] = sum[layer]/(len(models))

        return sum
            
        