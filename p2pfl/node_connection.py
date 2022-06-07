import socket
import threading
import logging
from p2pfl.command import *
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.utils.observer import Observable

########################
#    NodeConnection    #
########################

# COSAS:
#   - si casca la conexión no se trata de volver a conectar

class NodeConnection(threading.Thread, Observable):

    def __init__(self, parent_node, s, addr):
        threading.Thread.__init__(self)
        self.name = "node_connection-" + str(parent_node.get_addr()[0]) + ":" + str(parent_node.get_addr()[1]) + "-" + str(addr[0]) + ":" + str(addr[1])
        Observable.__init__(self)
        self.terminate_flag = threading.Event()
        self.socket = s
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.errors = 0
        self.addr = addr
        self.num_samples = 0
        self.param_bufffer = b""
        self.sending_model = False
        
        self.ready = None
        self.models_added = None

        self.tmp = 0

        self.comm_protocol = CommunicationProtocol({
            CommunicationProtocol.BEAT: Beat_cmd(None,None),
            CommunicationProtocol.STOP: Stop_cmd(None,self),
            CommunicationProtocol.CONN_TO: Conn_to_cmd(parent_node,None),
            CommunicationProtocol.START_LEARNING: Start_learning_cmd(parent_node,None),
            CommunicationProtocol.STOP_LEARNING: Stop_learning_cmd(parent_node,None),
            CommunicationProtocol.PARAMS: Params_cmd(parent_node,self),
            CommunicationProtocol.NUM_SAMPLES: Num_samples_cmd(None,self),
            CommunicationProtocol.READY: Ready_cmd(None,self)
        })

    def get_addr(self):
        return self.addr

    def set_ready_status(self,round, models_added):
        self.ready = round
        self.models_added = models_added

    def get_ready_status(self):
        return self.ready,self.models_added

    def stop(self,local=False):
        if not local:
            self.send(CommunicationProtocol.build_stop_msg())
        self.terminate_flag.set()

    def clear_buffer(self):
        self.param_bufffer = b""

    def is_sending_model(self):
        return self.sending_model

    def set_sending_model(self,flag):
        self.sending_model = flag

    def set_num_samples(self,num):
        self.num_samples = num

    def add_param_segment(self,data):
        self.param_bufffer = self.param_bufffer + data

    def get_params(self):
        return self.param_bufffer

    ################### 
    #    Main Loop    #  --> Recive and process messages
    ###################

    def run(self):
        self.socket.settimeout(SOCKET_TIEMOUT)
        overflow = 0
        buffer = b""
        while not self.terminate_flag.is_set():
            try:
                # Recive and process messages
                msg = b""
                if overflow == 0:
                    msg = self.socket.recv(BUFFER_SIZE)
                else:
                    msg = buffer + self.socket.recv(overflow) #alinear el colapso
                    buffer = b""
                    overflow = 0

                if msg!=b"":
                    #Check colapse
                    overflow = CommunicationProtocol.check_collapse(msg)
                    if overflow>0:
                        buffer = msg[overflow:]
                        msg = msg[:overflow]
                        logging.debug("{} (NodeConnection Run) Collapse detected: {}".format(self.get_addr(), msg))

                    # Process message and count errors
                    results = self.comm_protocol.process_message(msg)
                    errors = len(results) - sum(results)
                    # Add errors to the counter
                    if errors>0:
                        self.errors += errors
                        # If we have too many errors, we stop the connection
                        if self.errors >= MAX_ERRORS:
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
        
        #Down Connection
        logging.debug("Closed connection: {}".format(self.get_addr()))
        self.notify(self) # Notify the parent node
        self.socket.close()

    ##################
    #    Messages    # 
    ##################

    # Send a message to the other node. Message sending isnt guaranteed
    def send(self, data, model=False): 
        # Check if the connection is still alive
        if not self.terminate_flag.is_set():
            try:
                # If model is sending, we cant send a message
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


