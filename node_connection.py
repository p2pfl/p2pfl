import socket
import threading

BUFFER_SIZE = 1024

class NodeConnection(threading.Thread):
    def __init__(self, nodo_padre, socket):

        threading.Thread.__init__(self)

        self.running = True
        self.nodo_padre = nodo_padre
        self.socket = socket

    #Connection loop
    #   -ping periódico
    #   -procesado de peticiones
    def run(self):
        self.sock.settimeout(10.0) #REVISAR COMO VAN LOS TIMEOUTS X SI NECESITO CONTROLARLO EN CADA ITER DEL BUCLE

        while True:
            try:
                msg = self.socket.recv(BUFFER_SIZE)
                msg = str(msg.decode("utf-8"))

            except socket.timeout:
                print("NodeConnection Timeout")
                break

            except Exception as e:
                print("NodeConnection Timeout" + str(e))
                break
        
            self.__do_things(msg)
    
        #Bajamos Nodo
        self.nodo_padre.remove_node(self)
        self.sock.settimeout(None)
        self.sock.close()

    def __do_things(self, action):
        print(action + "chhejou")
        if action == "ping":
            pass
        elif action == "":
            pass
        else:
            print("Nao Comprendo")


    def send(self, data): pass
    def stop(self, data): pass