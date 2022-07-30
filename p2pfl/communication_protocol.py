import logging
from datetime import datetime
import random
import threading
from p2pfl.settings import Settings

###############################
#    CommunicationProtocol    # --> Invoker of Command Patern
###############################

class CommunicationProtocol:
    """
    Manages the meaning of node communication messages. Some messages contains a hash at end, it is used as a unique identifier for the message, 
    this kind of messages are gossiped to the entire network.
    
    The valid messages can be classified into gossiped and non-gossiped:
        Gossiped messages: 
            - BEAT <node> <HASH> 
            - START_LEARNING <rounds> <epoches> <HASH>
            - STOP_LEARNING <HASH>
            - VOTE_TRAIN_SET <node> (<node> <punct>)* VOTE_TRAIN_SET_CLOSE <HASH> 
            - METRICS <node> <round> <loss> <metric> <HASH> 

        Non Gossiped messages (communication over only 2 nodes):
            - CONNECT <ip> <port> <full> <force>
            - CONNECT_TO <ip> <port>
            - STOP 
            - PARAMS <data> \PARAMS
            - MODELS_READY <round> 
            - MODELS_AGREGATED <node>* MODELS_AGREGATED_CLOSE
            - MODEL_INITIALIZED

    Furthermore, all messages consists of encoded text (utf-8), with the exception of the `PARAMS` message, which contains serialized binaries.

    Non-static methods are used to process the different messages. Static methods are used to build messages and process only the `CONNECT` message (handshake). 
    
    Args:
        command_dict: Dictionary with the callbacks to execute at `process_message`.

    Attributes:
        command_dict: Dictionary with the callbacks to execute at `process_message`.
        last_messages: List of the last messages received.
    """

    BEAT = "BEAT"
    """
    Beat message header.
    """
    STOP = "STOP"           
    """
    Stop message header.
    """
    CONN = "CONNECT" 
    """
    Connection message header.
    """        
    CONN_TO = "CONNECT_TO"
    """
    Connection to message header.
    """
    START_LEARNING = "START_LEARNING"
    """
    Start learning message header.
    """
    STOP_LEARNING = "STOP_LEARNING"
    """
    Stop learning message header.
    """
    PARAMS = "PARAMS" # special case (binary) 
    """
    Parameters message header.
    """
    PARAMS_CLOSE = "\PARAMS" # special case (binary)
    """
    Parameters message closing.
    """
    MODELS_READY = "MODELS_READY"    
    """
    Models ready message header.
    """ 
    METRICS = "METRICS"
    """
    Metrics message header.
    """
    VOTE_TRAIN_SET = "VOTE_TRAIN_SET" 
    """
    Vote train set message header.
    """
    VOTE_TRAIN_SET_CLOSE = "\VOTE_TRAIN_SET"
    """
    Vote train set message closing.
    """
    MODELS_AGREGATED = "MODELS_AGREGATED"    
    """
    Models agregated message header.
    """
    MODELS_AGREGATED_CLOSE = "\MODELS_AGREGATED" 
    """
    Models agregated message closing.
    """
    MODEL_INITIALIZED = "MODEL_INITIALIZED" 
    """
    Model initialized message header.
    """
    
    ############################################
    #    MSG PROCESSING (Non Static Methods)   #
    ############################################

    def __init__(self, command_dict):
        self.command_dict = command_dict
        self.last_messages = []
        self.__last_messages_lock = threading.Lock()

    def add_processed_messages(self,messages):
        """
        Add messages to the last messages list. If ammount is higher than `Settings.AMOUNT_LAST_MESSAGES_SAVED` remove the oldest to keep the size.

        Args:
            hash_messages: List of hashes of the messages.
        """
        self.__last_messages_lock.acquire()
        self.last_messages = self.last_messages + messages
        # Remove oldest messages
        if len(self.last_messages)>Settings.AMOUNT_LAST_MESSAGES_SAVED:
            self.last_messages = self.last_messages[len(self.last_messages)-Settings.AMOUNT_LAST_MESSAGES_SAVED:]
        self.__last_messages_lock.release()

    def process_message(self, msg):
        """
        Processes messages and executes the callback associated with it (from ``command_dict``).        
        
        Args:
            msg: The message to process.

        Returns:
            tuple: (messages_executed, error) messages_executed is a list of the messages executed, error true if there was an error.

        """
        self.tmp_exec_msgs = {}
        error = False

        # Determine if is a binary message or not
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        if msg[0:len(header)] == header:
            end = CommunicationProtocol.PARAMS_CLOSE.encode("utf-8")

            # Check if done
            end_pos = msg.find(end)
            if end_pos != -1:
                return [], not self.__exec(CommunicationProtocol.PARAMS, None, None, msg[len(header):end_pos], True)

            return [],not self.__exec(CommunicationProtocol.PARAMS, None, None, msg[len(header):], False)

        else:      
            # Try to decode the message
            message = ""
            try:
                message = msg.decode("utf-8")
                message = message.split()
            except:
                error = True
        
            # Process messages
            while len(message) > 0:

                # Beat
                if message[0] == CommunicationProtocol.BEAT:
                    if len(message) > 2:
                        hash_ = message[2]
                        cmd_text = (" ".join(message[0:3]) + "\n").encode("utf-8")
                        if self.__exec(CommunicationProtocol.BEAT,hash_, cmd_text, message[1]):
                            message = message[3:]
                        else:
                            error = True
                            break
                    else:
                        error = True
                        break
                    
                # Stop (non gossiped)
                elif message[0] == CommunicationProtocol.STOP:
                    if self.__exec(CommunicationProtocol.STOP, None, None):
                        message = message[1:]
                    else:
                        error = True
                        break
                   

                # Connect to
                elif message[0] == CommunicationProtocol.CONN_TO:
                    if len(message) > 2:
                        if message[2].isdigit():
                            if self.__exec(CommunicationProtocol.CONN_TO, None, None, message[1], int(message[2])):
                                message = message[3:]
                            else:
                                error = True
                                break 
                        else:
                            error = True
                            break
                    else:
                        error = True
                        break

                # Start learning
                elif message[0] == CommunicationProtocol.START_LEARNING:
                    if len(message) > 3:
                        if message[1].isdigit() and message[2].isdigit():
                            hash_ = message[3]
                            cmd_text = (" ".join(message[0:4]) + "\n").encode("utf-8")
                            if self.__exec(CommunicationProtocol.START_LEARNING, hash_, cmd_text, int(message[1]), int(message[2])):
                                message = message[4:]
                            else:
                                error = True
                                break
                        else:
                            error = True
                            break
                    else:
                        error = True
                        break

                # Stop learning
                elif message[0] == CommunicationProtocol.STOP_LEARNING:
                    if len(message) > 1:            
                        if message[1].isdigit():
                            hash_ = message[1]
                            cmd_text = (" ".join(message[0:2]) + "\n").encode("utf-8")
                            if self.__exec(CommunicationProtocol.STOP_LEARNING, hash_, cmd_text):
                                message = message[2:]
                            else:
                                error = True
                                break
                        else:
                            error = True
                            break
                    else:
                        error = True
                        break
        
                # Models Ready
                elif message[0] == CommunicationProtocol.MODELS_READY:
                    if len(message) > 1:
                        if message[1].isdigit():
                            if self.__exec(CommunicationProtocol.MODELS_READY, None, None, int(message[1])):
                                message = message[2:]
                            else:
                                error = True
                                break
                        else:
                            error = True
                            break
                    else:
                        error = True
                        break

                # Metrics
                elif message[0] == CommunicationProtocol.METRICS:
                    if len(message) > 5:
                        try:
                            hash_ = message[5]
                            cmd_text = (" ".join(message[0:6]) + "\n").encode("utf-8")
                            if self.__exec(CommunicationProtocol.METRICS, hash_, cmd_text, message[1], int(message[2]), float(message[3]), float(message[4])):
                                message = message[6:]
                            else:
                                error = True
                                break
                        except Exception as e:
                            error = True
                            break
                    else:
                        error = True
                        break

                # Vote train set
                elif message[0] == CommunicationProtocol.VOTE_TRAIN_SET:
                    try:
                        # Divide messages and check length of message
                        close_pos = message.index(CommunicationProtocol.VOTE_TRAIN_SET_CLOSE)
                        node = message[1]
                        vote_msg = message[2:close_pos]
                        hash_ = message[close_pos+1]
                        cmd_text = (" ".join(message[0:close_pos+2]) + "\n").encode("utf-8")

                        if len(vote_msg)%2 != 0:
                            raise Exception("Invalid vote message")
                        message = message[close_pos+2:]

                        # Process vote message
                        votes = []
                        for i in range(0, len(vote_msg), 2):
                            votes.append((vote_msg[i], int(vote_msg[i+1])))

                        if not self.__exec(CommunicationProtocol.VOTE_TRAIN_SET, hash_, cmd_text, node,dict(votes)):
                            error = True
                            break

                    except Exception as e:
                        logging.exception(e)
                        error = True
                        break
                    
                # Models Agregated
                elif message[0] == CommunicationProtocol.MODELS_AGREGATED:
                    try:
                        # Divide messages and check length of message
                        close_pos = message.index(CommunicationProtocol.MODELS_AGREGATED_CLOSE)
                        content = message[1:close_pos]
                        message = message[close_pos+1:]

                        # Get Nodes
                        nodes=[]
                        for n in content:
                            nodes.append(n)

                        # Exec
                        if not self.__exec(CommunicationProtocol.MODELS_AGREGATED, None, None, nodes):
                            error = True
                            break

                    except Exception as e:
                        logging.exception(e)
                        error = True
                        break

                # Model Initialized
                elif message[0] == CommunicationProtocol.MODEL_INITIALIZED:
                    if self.__exec(CommunicationProtocol.MODEL_INITIALIZED, None, None):
                        message = message[1:]
                    else:
                        error = True
                        break
                    
                # Non Recognized message            
                else:
                    error = True
                    break
                
            # Return
            return self.tmp_exec_msgs,error

    # Exec callbacks
    def __exec(self,action,hash_, cmd_text, *args):
        try:
            # Check if can be executed
            if hash_ not in self.last_messages or hash_ is None:
                self.command_dict[action].execute(*args)
                # Save to gossip
                if hash_ is not None:
                    self.tmp_exec_msgs[hash_] = cmd_text
                    self.add_processed_messages([hash_])
                return True
            return True
        except Exception as e:
            logging.info("Error executing callback: " + str(e))
            logging.exception(e)
            return False


    ########################################
    #    MSG PROCESSING (Static Methods)   #
    ########################################

    def process_connection(message, callback):
        """"
        Static method that checks if the message is a valid connection message and executes the callback (accept connection).

        Args:
            message: The message to check.
            callback: What do if the connection message is legit.
        
        Returns:
            True if connection was accepted, False otherwise.
        """
        message = message.split()
        if len(message) > 4:
            if message[0] == CommunicationProtocol.CONN:
                try:
                    full = message[3] == "1"
                    force = message[4] == "1"
                    callback(message[1], int(message[2]), full, force)
                    return True
                except:
                    return False
            else:
                return False
        else:
            return False

    def check_collapse(msg):
        """"
        Static method that checks if in the message there is a collapse (a binary message (it should fill all the buffer) and a non-binary message before it).

        Actually, collapses can only happen with ``PARAMS`` binary message.

        Args:
            msg: The message to check.

        Returns:
            Length of the collapse (number of bytes to the binary headear). 
        """
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        header_pos = msg.find(header)
        if header_pos != -1 and msg[0:len(header)] != header:
            return header_pos
   
        return 0

    def check_params_incomplete(msg):
        """
        Checks if a params message is incomplete. If the message is complete or is not a params message, it returns 0.
        
        Returns:
            Number of bytes that needs to be complete
        """
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        if msg[0:len(header)] == header:
            if len(msg)<Settings.BLOCK_SIZE:
                return Settings.BLOCK_SIZE-len(msg)

        return 0
    

    #######################################
    #     MSG BUILDERS (Static Methods)   #
    #######################################

    def generate_hased_message(msg):
        """
        Static method that given a non-encoded message generates a hashed and encoded message.

        Args:
            msg: Non encoded message.

        Returns:
            Hashed and encoded message.
        """
        # random number to avoid generating the same hash for a different message (at she same time)
        id = hash(msg+str(datetime.now())+str(random.randint(0,100000)))
        return (msg + " " + str(id) + "\n").encode("utf-8")

    def build_beat_msg(node):
        """ 
        Returns:
            A encoded beat message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.BEAT + " " + node)

    def build_stop_msg():
        """ 
        Returns:
            A encoded stop message.
        """
        return (CommunicationProtocol.STOP + "\n").encode("utf-8")

    def build_connect_to_msg(ip, port):
        """
        Args:
            ip: The ip address to connect to.
            port: The port to connect to.

        Returns:
            A encoded connect to message.
        """
        return (CommunicationProtocol.CONN_TO + " " + ip + " " + str(port) + "\n").encode("utf-8")

    def build_start_learning_msg(rounds, epochs):
        """
        Args:
            rounds: The number of rounds to train.
            epochs: The number of epochs to train.

        Returns:
            A encoded start learning message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.START_LEARNING + " " + str(rounds) + " " + str(epochs))

    def build_stop_learning_msg():
        """
        Returns:
            A encoded stop learning message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.STOP_LEARNING)

    def build_models_ready_msg(round):
        """
        Args:
            round: The last round finished.

        Returns:
            A encoded ready message.
        """
        return (CommunicationProtocol.MODELS_READY + " " + str(round) + "\n").encode("utf-8")

    def build_metrics_msg(node, round, loss, metric):
        """
        Args:
            node: The node that sent the message.
            round: The round when the metrics was calculated.
            loss: The loss of the last round.
            metric: The metric of the last round.
        
        Returns:
            A encoded metrics message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.METRICS + " " + node + " " + str(round) + " " + str(loss) + " " + str(metric))

    def build_vote_train_set_msg(node,votes):
        """
        Args:
            node: The node that sent the message.
            votes: Votes of the node.
        
        Returns:
            A encoded vote train set message.
        """
        aux = ""
        for v in votes:
            aux = aux + " " + v[0]+ " " + str(v[1])
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.VOTE_TRAIN_SET + " " + node + aux + " " + CommunicationProtocol.VOTE_TRAIN_SET_CLOSE)
        
    def build_models_agregated_msg(nodes):
        """
        Args:
            nodes: List of strings to indicate agregated nodes.
        
        Returns:
            A encoded models agregated message.
        """
        aux = ""
        for n in nodes:
            aux = aux + " " + n
        return (CommunicationProtocol.MODELS_AGREGATED + aux + " " + CommunicationProtocol.MODELS_AGREGATED_CLOSE + "\n").encode("utf-8")

    def build_model_initialized_msg():
        """
        Returns:
            A encoded model inicialized message.
        """
        return (CommunicationProtocol.MODEL_INITIALIZED + "\n").encode("utf-8")

    def build_connect_msg(ip, port, broadcast, force):
        """
        Build Handshake message.
        Not Hashed. Special case of message.

        Args:
            ip: The ip address of the node that tries to connect.
            port: The port of the node that tries to connect.
            broadcast: Whether or not to broadcast the message.
            force: Whether or not to force connection.

        Returns:
            A encoded connect message.
        """
        return (CommunicationProtocol.CONN + " " + ip + " " + str(port) + " " + str(broadcast) + " " + str(force) + "\n").encode("utf-8")

    def build_params_msg(data):
        """
        Build model serialized messages.
        Not Hashed. Special case of message (binary message).

        Args:
            data: The model parameters to send (encoded).

        Returns:
            A list of fragments messages of the params.
        """
        # Encoding Headers and ending
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        end = CommunicationProtocol.PARAMS_CLOSE.encode("utf-8")

        # Spliting data
        size = Settings.BLOCK_SIZE - len(header)
        data_msgs = []
        for i in range(0, len(data), size):
            data_msgs.append(header + (data[i:i+size]))

        # Adding closing message
        if len(data_msgs[-1]) + len(end) <= Settings.BLOCK_SIZE:
            data_msgs[-1] += end
            data_msgs[-1] += b'\0' * (Settings.BLOCK_SIZE - len(data_msgs[-1])) # padding to avoid message fragmentation
        else:
            data_msgs.append(header + end)

        return data_msgs