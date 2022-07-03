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
    Manages the meaning of communication messages. The valid messages are:
        Gossiped messages: 
            - BEAT <HASH> -----------------------------------------------------------------cambiar (indicar el nodo que lo envio)
            - START_LEARNING <rounds> <epoches> <HASH>
            - STOP_LEARNING <HASH>
            - METRICS <round> <loss> <metric> <HASH> -----------------------------------------------------------------cambiar (indicar el nodo que lo envio)

        Non Gossiped messages (communication over only 2 nodes):
            - CONNECT <ip> <port> <broadcast>
            - CONNECT_TO <ip> <port>
            - STOP 
            - NUM_SAMPLES <train_num> <test_num>
            - PARAMS <data> \PARAMS
            - MODELS_READY <round> -----------------------------------------------------------------cambiar con heartbeater 2.0
            - VOTE_TRAIN_SET (<node> <punct>)* VOTE_TRAIN_SET_CLOSE -----------------------------------------------------------------cambiar con heartbeater 2.0
            - LEARNING_IS_RUNNING <round> <total_rounds>
            - MODELS_AGREGATED <node>* MODELS_AGREGATED_CLOSE

    The unique non-static method is used to process messages with a connection stablished. 
    
    XXXXXXXXXXXXXXXXXXXXXX EXPLAIN GOSSIP

    Args:
        command_dict: Dictionary with the callbacks to execute at `process_message`.

    Attributes:
        command_dict: Dictionary with the callbacks to execute at `process_message`.
    """

    BEAT                    = "BEAT"
    STOP                    = "STOP"            # Non Gossiped
    CONN                    = "CONNECT"         # Non Gossiped
    CONN_TO                 = "CONNECT_TO"
    START_LEARNING          = "START_LEARNING"
    STOP_LEARNING           = "STOP_LEARNING"
    NUM_SAMPLES             = "NUM_SAMPLES"     # Non Gossiped
    PARAMS                  = "PARAMS"          # special case (binary) 
    PARAMS_CLOSE            = "\PARAMS"         # special case (binary)
    MODELS_READY            = "MODELS_READY"    
    METRICS                 = "METRICS"
    VOTE_TRAIN_SET          = "VOTE_TRAIN_SET" # ---------- cambiarlo por node (h:p) en vez de h p
    VOTE_TRAIN_SET_CLOSE    = "\VOTE_TRAIN_SET"
    LEARNING_IS_RUNNING     = "LEARNING_IS_RUNNING" # Non Gossiped
    MODELS_AGREGATED        = "MODELS_AGREGATED"    # Non Gossiped
    MODELS_AGREGATED_CLOSE  = "\MODELS_AGREGATED"    # Non Gossiped


    ############################################
    #    MSG PROCESSING (Non Static Methods)   #
    ############################################


    def __init__(self, command_dict):
        self.command_dict = command_dict
        self.last_messages = []
        self.last_messages_lock = threading.Lock()

    def add_processed_messages(self,messages):
        """
        Add messages to the last messages. If ammount is higher than `Settings.AMOUNT_LAST_MESSAGES_SAVED` remove the oldest to keep the size.
        Args:
            hash_messages: List of hashes of the messages.
        """
        self.last_messages_lock.acquire()
        self.last_messages = self.last_messages + messages
        # Remove oldest messages
        if len(self.last_messages)>Settings.AMOUNT_LAST_MESSAGES_SAVED:
            self.last_messages = self.last_messages[len(self.last_messages)-Settings.AMOUNT_LAST_MESSAGES_SAVED:]
        self.last_messages_lock.release()

    def process_message(self, msg):
        """
        Processes a message and executes the callback associated with it.        
        
        Args:
            msg: The message to process.

        Returns:
            True if the message was processed and no errors occurred, False otherwise.

        """
        self.tmp_exec_msgs = {}
        error = False
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
                # Check message and exec message
                if len(message) > 0:
                    # Beat
                    if message[0] == CommunicationProtocol.BEAT:
                        if len(message) > 1:
                            hash_ = message[1]
                            cmd_text = (" ".join(message[0:2]) + "\n").encode("utf-8")
                            if self.__exec(CommunicationProtocol.BEAT,hash_, cmd_text):
                                message = message[2:]
                            else:
                                error = True
                                break
                        else:
                            error = True
                            break
                    
                    # Stop (non gossiped)
                    elif message[0] == CommunicationProtocol.STOP:
                        if len(message) > 0:
                            if self.__exec(CommunicationProtocol.STOP, None, None):
                                message = message[1:]
                            else:
                                error = True
                                break
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
        
                    # Number of samples (non gossiped)
                    elif message[0] == CommunicationProtocol.NUM_SAMPLES:
                        if len(message) > 2:
                            if message[1].isdigit() and message[2].isdigit():
                                if self.__exec(CommunicationProtocol.NUM_SAMPLES, None, None, int(message[1]), int(message[2])):
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
                        if len(message) > 4:
                            try:
                                hash_ = message[4]
                                cmd_text = (" ".join(message[0:5]) + "\n").encode("utf-8")
                                if self.__exec(CommunicationProtocol.METRICS, hash_, cmd_text, int(message[1]), float(message[2]), float(message[3])):
                                    message = message[5:]
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
                            vote_msg = message[1:close_pos]
                            if len(vote_msg)%2 != 0:
                                raise Exception("Invalid vote message")
                            message = message[close_pos+1:]

                            # Process vote message
                            votes = []
                            for i in range(0, len(vote_msg), 2):
                                votes.append((vote_msg[i], int(vote_msg[i+1])))

                            if not self.__exec(CommunicationProtocol.VOTE_TRAIN_SET, None, None, dict(votes)):
                                error = True
                                break

                        except Exception as e:
                            error = True
                            break

                    # Learning is running
                    elif message[0] == CommunicationProtocol.LEARNING_IS_RUNNING:
                        if len(message) > 2:
                            if message[1].isdigit() and message[2].isdigit() and message[3].isdigit():
                                if self.__exec(hash_,CommunicationProtocol.LEARNING_IS_RUNNING, None, None, int(message[1])):
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
                                host,port = n.split(":")
                                port = int(port)
                                nodes.append((host,port))

                            # Exec
                            if not self.__exec(CommunicationProtocol.MODELS_AGREGATED, None, None, nodes):
                                error = True
                                break

                        except Exception as e:
                            logging.exception(e)
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
                    self.add_processed_messages([hash_]) # si es una única instancia llegaría con esto
                return True
            return True
        except Exception as e:
            logging.info("Error executing callback: " + str(e))
            logging.exception(e)
            return False


    ########################################
    #    MSG PROCESSING (Static Methods)   #
    ########################################


    # Check if connection is correct and execute the callback (static)
    def process_connection(message, callback):
        """"
        Static method that checks if the message is a valid connection message and executes the callback (do the connection).

        Args:
            message: The message to check.
            callback: What do if the connection message is legit.
        
        """
        message = message.split()
        if len(message) > 3:
            if message[0] == CommunicationProtocol.CONN:
                try:
                    broadcast = message[3] == "1"
                    callback(message[1], int(message[2]), broadcast)
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
    

    #######################################
    #     MSG BUILDERS (Static Methods)   #
    #######################################


    def generate_hased_message(msg):
        # random number to avoid generating the same hash for a different message (at she same time)
        id = hash(msg+str(datetime.now())+str(random.randint(0,100000)))
        return (msg + " " + str(id) + "\n").encode("utf-8")

    def build_beat_msg():
        """ 
        Returns:
            A encoded beat message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.BEAT)

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


    def build_num_samples_msg(num):
        """
        Args:
            num: (Tuple) The number of samples to train and test.

        Returns:
            A encoded number of samples message.
        """
        return (CommunicationProtocol.NUM_SAMPLES + " " + str(num[0]) + " " + str(num[1]) + "\n").encode("utf-8")


    def build_models_ready_msg(round):
        """
        Args:
            round: The last round finished.

        Returns:
            A encoded ready message.
        """
        return (CommunicationProtocol.MODELS_READY + " " + str(round) + "\n").encode("utf-8")


    def build_metrics_msg(round, loss, metric):
        """
        Args:
            round: The round when the metrics was calculated.
            loss: The loss of the last round.
            metric: The metric of the last round.
        
        Returns:
            A encoded metrics message.
        """
        return CommunicationProtocol.generate_hased_message(CommunicationProtocol.METRICS + " " + str(round) + " " + str(loss) + " " + str(metric))


    def build_vote_train_set_msg(votes):
        """
        Args:
            candidates: The candidates to vote for.
            weights: The weights of the candidates.
        
        Returns:
            A encoded vote train set message.
        """
        aux = ""
        for v in votes:
            aux = aux + " " + v[0]+ " " + str(v[1])
        return (CommunicationProtocol.VOTE_TRAIN_SET + aux + " " + CommunicationProtocol.VOTE_TRAIN_SET_CLOSE + "\n").encode("utf-8")
        

    def build_learning_is_running_msg(round, epoch):
        """
        Args:
            round: The round that is running.
            epoch: The epoch that is running.
        
        Returns:
            A encoded learning is running message.
        """
        return (CommunicationProtocol.LEARNING_IS_RUNNING + " " + str(round) + " " + str(epoch) + "\n").encode("utf-8")

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


    ###########################
    #     Special Messages    #
    ###########################

    def build_connect_msg(ip, port, broadcast):
        """
        Build Handshake message.

        Not Hashed. Special case of message.

        Args:
            ip: The ip address of the node that tries to connect.
            port: The port of the node that tries to connect.
            broadcast: Whether or not to broadcast the message.

        Returns:
            A encoded connect message.
        """
        return (CommunicationProtocol.CONN + " " + ip + " " + str(port) + " " + str(broadcast) + "\n").encode("utf-8")

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
        size = Settings.BUFFER_SIZE - len(header)
        data_msgs = []
        for i in range(0, len(data), size):
            data_msgs.append(header + (data[i:i+size]))

        # Adding closing message
        if len(data_msgs[-1]) + len(end) <= Settings.BUFFER_SIZE:
            data_msgs[-1] += end
            data_msgs[-1] += b'\0' * (Settings.BUFFER_SIZE - len(data_msgs[-1])) # padding to avoid message fragmentation
        else:
            data_msgs.append(header + end)

        return data_msgs