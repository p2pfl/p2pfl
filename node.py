import socket
import threading
import logging
import time

#USAR UN LOGGER PARA PRINTS MÁS BONITOS

#XQ SOCKETS? -> received at any time (async)

#Si fuese pa montar una red distribuida elixir de 1

#socket no bloqueantes? -> select

# https://github.com/GianisTsol/python-p2p/blob/master/pythonp2p/node.py


BUFFER_SIZE = 1024


#que extienda de thread?
class Node(threading.Thread):
    def __init__(self, host, port):

        threading.Thread.__init__(self)

        self.running = True
        self.host = host
        self.port = port

        # Setting Up Node Socket (listening)
        self.node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP
        #self.node_socket.settimeout(0.2)
        self.node_socket.bind((host, port))
        
        self.node_socket.listen(5)# no mas de 5 peticones a la cola

        # Neightboors
        self.neightboors = []


    #Objetivo: Agregar vecinos a la lista -> CREAR POSIBLES SOCKETS 
    def run(self):
        logging.info('Nodo a la escucha en {} {}'.format(self.host, self.port))
        while True: #meterle threading flags
            try:
                (node_socket, address) = self.node_socket.accept()
                logging.info('Conexión aceptada en {}'.format(address))
                
                msg = node_socket.recv(BUFFER_SIZE) 
                msg = msg.decode("UTF-8")
                print(msg)
                if msg == "hola":
                    print("Te anadiremos a la lista de vecinos")
                else:
                        node_socket.close()
           
            except Exception as e:
                #revisar excepciones de timeout y configurarlas
                print(e)

        #Detenemos nodo
        self.pinger.stop()
        for n in self.neightboors:
            n.stop()

        self.node_socket.close()

    def send(self, data): pass
    def stop(self, data): pass
    def connect_to(self,node): pass    

    def broadcast(self, msg, exc=None):
        for n in self.neightboors:
            if n != exc:
                print("Not implemented yet")

    #############################
    #  Neighborhood management  #
    #############################

    def get_neighbors(self):pass
    def add_neighbor(self):pass
    def rm_neighbor(self):pass

  


node = Node("localhost",6666)
