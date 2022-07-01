import socket
import threading
import logging
from p2pfl.command import *
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.encrypter import Encrypter
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

########################
#    NodeConnection    #
########################

# organizar algo código

class NodeConnection(threading.Thread, Observable):
    """
    This class represents a connection to a node. It is a thread, so it's going to process all messages in a background thread using the CommunicationProtocol.

    The NodeConnection can recive many messages in a single recv and exists 2 kinds of messages:
        - Binary messages (models)
        - Text messages (commands)

    Carefully, if the connection is broken, it will be closed. If the user wants to reconnect, he/she should create a new connection.

    Args:
        parent_node: The parent node of this connection.
        s: The socket of the connection.
        addr: The address of the node that is connected to.
    """

    def __init__(self, parent_node_name, s, addr):
        threading.Thread.__init__(self)
        self.name = "node_connection-" + parent_node_name + "-" + str(addr[0]) + ":" + str(addr[1])
        Observable.__init__(self)
        self.terminate_flag = threading.Event()
        self.socket = s
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.errors = 0
        self.addr = addr
        self.train_num_samples = 0
        self.test_num_samples = 0
        self.param_bufffer = b""
        self.sending_model = False
        
        self.model_ready = -1

        self.train_set_votes = []

        self.comm_protocol = CommunicationProtocol({
            CommunicationProtocol.BEAT: Beat_cmd(self),
            CommunicationProtocol.STOP: Stop_cmd(self),
            CommunicationProtocol.CONN_TO: Conn_to_cmd(self),
            CommunicationProtocol.START_LEARNING: Start_learning_cmd(self),
            CommunicationProtocol.STOP_LEARNING: Stop_learning_cmd(self),
            CommunicationProtocol.NUM_SAMPLES: Num_samples_cmd(self),
            CommunicationProtocol.PARAMS: Params_cmd(self),
            CommunicationProtocol.MODELS_READY: Models_Ready_cmd(self),
            CommunicationProtocol.METRICS: Metrics_cmd(self),
            CommunicationProtocol.VOTE_TRAIN_SET: Vote_train_set_cmd(self),
            CommunicationProtocol.LEARNING_IS_RUNNING: Learning_is_running_cmd(self),
        })

    def add_processed_messages(self,msgs):
        """
        Add to a list of messages that have been processed. (By other nodes)

        Args:
            msgs: The list of messages that have been processed.

        """
        self.comm_protocol.add_processed_messages(msgs)

    def get_addr(self):
        """
        Returns:
            The address of the node that is connected to.   
        """
        return self.addr

    def set_model_ready_status(self,round):
        """
        Set the last ready round of the other node.

        Args:
            round: The last ready round of the other node.
        """
        self.model_ready = round
        self.notify(Events.NODE_MODELS_READY_EVENT, self)

    ###################
    # Train set votes #
    ###################

    def set_train_set_votes(self,votes):
        """
        Set the last ready round of the other node.

        Args:
            round: The last ready round of the other node.
        """
        self.train_set_votes = votes
        self.notify(Events.TRAIN_SET_VOTE_RECEIVED_EVENT, self)

    def get_train_set_votes(self):
        """
        Returns:
            The votes for the treining set of the other node.
        """
        return self.train_set_votes

    def clear_train_set_votes(self):
        """
        Clear the votes.
        """
        self.train_set_votes = []

    def get_ready_model_status(self):
        """
        Returns:
            The last ready round of the other node.
        """
        return self.model_ready

    def set_sending_model(self,flag):
        """
        Set when the model is being sent in the connection. (high bandwidth)
        
        Args:
            flag: True if the model is being sent, false otherwise.
        """
        self.sending_model = flag

    def is_sending_model(self):
        """
        Returns:    
            True if the model is being sent, False otherwise.
        """
        return self.sending_model

    def set_num_samples(self,train,test):
        """
        Indicates the number of samples of the otrh node.
         
        Args:
            num: The number of samples of the other node.
        """
        self.train_num_samples = train
        self.test_num_samples = test

    def add_param_segment(self,data):
        """
        Add a segment of parameters to the buffer.
        
        Args:
            data: The segment of parameters.
        """
        self.param_bufffer = self.param_bufffer + data

    def get_params(self):
        """
        Returns:
            The parameters buffer content.
        """
        return self.param_bufffer

    def clear_buffer(self):
        """
        Clear the params buffer.
        """
        self.param_bufffer = b""

    ################### 
    #    Main Loop    # 
    ###################

    def start(self):
        self.notify(Events.NODE_CONNECTED_EVENT, self)
        return super().start()


    def run(self):
        """
        NodeConnection loop. Recive and process messages.
        """
        self.socket.settimeout(Settings.SOCKET_TIEMOUT)
        overflow = 0
        buffer = b""
        while not self.terminate_flag.is_set():
            try:
                # Recive message
                msg = b""
                if overflow == 0:
                    msg = self.socket.recv(Settings.BUFFER_SIZE)
                else:
                    msg = buffer + self.socket.recv(overflow) #alinear el colapso
                    buffer = b""
                    overflow = 0
                
                # Process messages
                if msg!=b"":
                    #Check if colapse is happening
                    overflow = CommunicationProtocol.check_collapse(msg)
                    if overflow>0:
                        buffer = msg[overflow:]
                        msg = msg[:overflow]
                        logging.debug("{} (NodeConnection Run) Collapse detected: {}".format(self.get_addr(), msg))

                    # Process message and count errors
                    exec_msgs,error = self.comm_protocol.process_message(msg)
                    if len(exec_msgs) > 0:
                        self.notify(Events.PROCESSED_MESSAGES_EVENT, (self,exec_msgs)) # Notify the parent node

                    # Error happened
                    if error:
                        self.terminate_flag.set()
                        logging.debug("An error happened. {}".format(self.get_addr()))
                        logging.debug("Last error: {}".format(msg))           

            except socket.timeout:
                logging.debug("{} (NodeConnection Loop) Timeout".format(self.get_addr()))
                self.terminate_flag.set()
                break

            except Exception as e:
                logging.debug("{} (NodeConnection Loop) Exception: ".format(self.get_addr()) + str(e))
                self.terminate_flag.set()
                break
        
        #Down Connection
        logging.debug("Closed connection: {}".format(self.get_addr()))
        self.notify(Events.END_CONNECTION, self) # Notify the parent node
        self.socket.close()

    def stop(self,local=False):
        """
        Stop the connection.
        
        Args:
            local: If true, the connection will be closed without notifying the other node.
        """
        if not local:
            self.send(CommunicationProtocol.build_stop_msg())
        self.terminate_flag.set()

    ##################
    #    Messages    # 
    ##################

    # Send a message to the other node. Message sending isnt guaranteed
    def send(self, data, is_necesary=True): 
        """
        Send a message to the other node.

        Args:
            data: The message to send.
            is_necesary: If true, the message is guaranteed to be sent.

        Returns:
            True if the message was sent, False otherwise.

        """    
        # Check if the connection is still alive
        if not self.terminate_flag.is_set():
            try:
                # If model is sending, we cant send a message
                if not self.is_sending_model() or is_necesary:
                    self.socket.sendall(data)
                    return True
                else:
                    return False
            except Exception as e:
                logging.debug("{} (NodeConnection Send) Exception: ".format(self.get_addr()) + str(e))
                logging.exception(e)
                self.terminate_flag.set() #exit
                return False
        else:
            return False

    ###########################
    #    Command Callbacks    #
    ###########################

    def notify_conn_to(self, h, p):
        """
        Notify to the parent node that `CONN_TO` has been received.
        """
        self.notify(Events.CONN_TO, (h,p))

    def notify_start_learning(self, r, e):
        """
        Notify to the parent node that `START_LEARNING` has been received.
        """
        self.notify(Events.START_LEARNING, (r,e))

    def notify_stop_learning(self,cmd):
        """
        Notify to the parent node that `START_LEARNING` has been received.
        """
        self.notify(Events.STOP_LEARNING, None)

    def notify_params(self,params):
        """
        Notify to the parent node that `PARAMS` has been received.
        """
        self.notify(Events.PARAMS_RECEIVED, (str(self.get_addr()), params, self.train_num_samples))

    def notify_metrics(self,round,loss,metric):
        """
        Notify to the parent node that `METRICS` has been received.
        """
        name = str(self.get_addr()[0]) + ":" + str(self.get_addr()[1])
        self.notify(Events.METRICS_RECEIVED, (name, round, loss, metric))

    def notify_learning_is_running(self,round,total_rounds):
        """
        Notify to the parent node that `LEARNING_IS_RUNNING` has been received.
        """
        self.notify(Events.LEARNING_IS_RUNNING_EVENT, (round,total_rounds))