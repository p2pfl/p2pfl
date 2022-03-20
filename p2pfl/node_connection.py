import socket
import threading
import logging
import time

BUFFER_SIZE = 1024


#Pulir tema de direcciones, fijo que se pueden sacar de socket
class NodeConnection(threading.Thread):
    def __init__(self, nodo_padre, socket, buffer, addr):

        threading.Thread.__init__(self)

        self.terminate_flag = threading.Event()
        self.nodo_padre = nodo_padre
        self.addr = addr
        self.socket = socket
        self.buffer = buffer


    def get_addr(self):
        return self.addr

    #Connection loop
    #   -ping periódico
    #   -procesado de peticiones
    def run(self):
        self.socket.settimeout(30) #REVISAR COMO VAN LOS TIMEOUTS X SI NECESITO CONTROLARLO EN CADA ITER DEL BUCLE

        while not self.terminate_flag.is_set():
            try:
                msg = self.socket.recv(BUFFER_SIZE)
                msg = str(msg.decode("utf-8"))
                
            except socket.timeout:
                logging.debug("{} NodeConnection Timeout".format(self.addr))
                self.stop()
                break

            except Exception as e:
                logging.debug("{} Exception: ".format(self.addr) + str(e))
                self.stop()
                break


            #Si no hace nada espera 1
            if not self.__do_things(msg):
                time.sleep(1)   
                print("----1----")

        
        #Bajamos Nodo
        print("bajamos nodo!")
        self.nodo_padre.rm_neighbor(self)
        logging.debug("Closed connection with {}".format(self.addr))
        self.socket.close()

    def __do_things(self, action):
        #logging.debug("{} {}".format(self.addr,action))


        if action == "ping":
            logging.debug("Ping {}".format(self.addr))

        elif action == "":
            return False
        else:
            print("Nao Comprendo (" + action + ")")


    def send(self, data): 
        self.socket.sendall(data)

    # No es un stop instantáneo
    def stop(self):
        self.terminate_flag.set()

