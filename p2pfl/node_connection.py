from asyncio.log import logger
import socket
from statistics import mode
import threading
import logging
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *


# METER CONTROL DE ESTADOS PARA EVITAR ATAQUES
# ENMASCARR SOCKET EN CLASE COMUNICATIONS SYSTEM

# COSAS:
#   - si casca la conexión no se trata de volver a conectar


class NodeConnection(threading.Thread):

    def __init__(self, nodo_padre, socket, addr):
        threading.Thread.__init__(self)
        self.terminate_flag = threading.Event()
        self.nodo_padre = nodo_padre
        self.socket = socket
        self.errors = 0
        self.addr = addr
        self.param_bufffer = b""
        self.sending_model = False
        self.comm_protocol = CommunicationProtocol({
            CommunicationProtocol.BEAT: self.__on_beat,
            CommunicationProtocol.STOP: self.__on_stop,
            CommunicationProtocol.CONN_TO: self.__on_conn_to,
            CommunicationProtocol.START_LEARNING: self.__on_start_learning,
            CommunicationProtocol.STOP_LEARNING: self.__on_stop_learning,
            CommunicationProtocol.PARAMS: self.__on_params,
        })


    def run(self):
        self.socket.settimeout(TIEMOUT)

        while not self.terminate_flag.is_set():
            try:
                # Recive and process messages
                msg = self.socket.recv(BUFFER_SIZE)
                if msg!=b"":
                    if not self.comm_protocol.process_message(msg):
                        self.errors += 1
                        # If we have too many errors, we stop the connection
                        if self.errors > 1:#10:
                            self.terminate_flag.set()
                            logging.debug("Too mucho errors. {}".format(self.get_addr()))
                            logging.debug("Last error: {}".format(msg))           

            except socket.timeout:
                logging.debug("{} (NodeConnection Loop) Timeout".format(self.get_addr()))
                self.terminate_flag.set()
                break

            except Exception as e:
                logging.debug("{} (NodeConnection Loop) Exception: ".format(self.get_addr()) + str(e))
                #logging.exception(e)
                self.terminate_flag.set()
                break
        
        #Bajamos Nodo
        logging.debug("Closed connection: {}".format(self.get_addr()))
        self.nodo_padre.rm_neighbor(self)
        self.socket.close()

    def get_addr(self):
        return self.addr

    # Envía un mensaje
    #   No garantiza envíos, debe ser el usuario el que se cerciore
    def send(self, data, model=False): 
        # Verificamos que el nodo esté operativo
        if not self.terminate_flag.is_set():
            try:
                # Si se está enviando el modelo no se podrá enviar un mensaje
                if not self.is_sending_model() or model:
                    self.socket.sendall(data)
                    return True
                else:
                    return False
            except Exception as e:
                logging.debug("{} (NodeConnection Send) Exception: ".format(self.get_addr()) + str(e))
                self.terminate_flag.set() #exit
                return False
        else:
            return False


    def stop(self):
        self.send(CommunicationProtocol.build_stop_msg())
        self.terminate_flag.set()

    def clear_buffer(self):
        self.param_bufffer = b""

    def is_sending_model(self):
        return self.sending_model

    def set_sending_model(self,flag):
        self.sending_model = flag

    #########################
    #       Callbacks       #
    #########################

    def __on_beat(self):
        pass #logging.debug("Beat {}".format(self.get_addr()))

    def __on_stop(self):
            self.terminate_flag.set()
           
    def __on_conn_to(self,h,p):
            self.nodo_padre.connect_to(h, p, full=False)

    def __on_start_learning(self, rounds, epochs):
        # creamos proceso para no bloqeuar la recepcion de mensajes
        learning_thread = threading.Thread(target=self.nodo_padre.start_learning,args=(rounds,epochs))
        learning_thread.start()

    def __on_stop_learning(self):
        self.nodo_padre.stop_learning()

    def __on_params(self,msg,done):
        if done:

            params = self.param_bufffer + msg
            self.clear_buffer()

            # ESTO ESTÁ MAL XQ SE VA A EJECUTAR DESDE EL HILO DE RECEPCIÓN
            self.nodo_padre.add_model(params)
            

        else:
            self.param_bufffer = self.param_bufffer + msg

