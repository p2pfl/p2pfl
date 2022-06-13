import logging
from p2pfl.const import BUFFER_SIZE

###############################
#    CommunicationProtocol    # --> Patrón commando 
###############################


class CommunicationProtocol:
    """
    Manages the meaning of communication messages. The valid messages are: 
        - BEAT
        - STOP
        - CONNECT <ip> <port> <broadcast>
        - CONNECT_TO <ip> <port>
        - START_LEARNING <rounds> <epoches>
        - STOP_LEARNING
        - NUM_SAMPLES <num> 
        - PARAMS <data> \PARAMS
        - READY <round>
        - SEND_MODEL

    The unique non-static method is used to process messages with a connection stablished.

    Args:
        command_dict: Dictionary with the callbacks to execute at `process_message`.

    Attributes:
        command_dict: Dictionary with the callbacks to execute at `process_message`.
    """

    BEAT           = "BEAT"
    STOP           = "STOP"
    CONN           = "CONNECT"
    CONN_TO        = "CONNECT_TO"
    START_LEARNING = "START_LEARNING"
    STOP_LEARNING  = "STOP_LEARNING"
    NUM_SAMPLES    = "NUM_SAMPLES"
    PARAMS         = "PARAMS" #special case
    PARAMS_CLOSE   = "\PARAMS" #special case
    READY          = "READY"
    SEND_MODEL     = "SEND_MODEL"

    ########################
    #    MSG PROCESSING    #
    ########################

    def __init__(self, command_dict):
        self.command_dict = command_dict
        self.__cmds_success = []

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
    
    def process_message(self, msg):
        """
        Processes a message and executes the callback associated with it.        
        
        Args:
            msg: The message to process.

        Returns:
            True if the message was processed and no errors occurred, False otherwise.

        """
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        if msg[0:len(header)] == header:
            end = CommunicationProtocol.PARAMS_CLOSE.encode("utf-8")

            # Check if done
            end_pos = msg.find(end)
            if end_pos != -1:
                return [self.__exec(CommunicationProtocol.PARAMS, msg[len(header):end_pos], True)]

            return [self.__exec(CommunicationProtocol.PARAMS, msg[len(header):], False)]

        else:      

            # Try to decode the message
            message = ""
            try:
                message = msg.decode("utf-8")
                message = message.split()
            except:
                self.__cmds_success.append(False)

            # Process messages
            while len(message) > 0:
                # Check message and exec message
                if len(message) > 0:
                    # Beat
                    if message[0] == CommunicationProtocol.BEAT:
                        self.__cmds_success.append(self.__exec(CommunicationProtocol.BEAT))
                        message = message[1:]

                    # Stop
                    elif message[0] == CommunicationProtocol.STOP:
                        self.__cmds_success.append(self.__exec(CommunicationProtocol.STOP))
                        message = message[1:] 

                    # Connect to
                    elif message[0] == CommunicationProtocol.CONN_TO:
                        if len(message) > 2:
                            if message[2].isdigit():
                                self.__cmds_success.append(self.__exec(CommunicationProtocol.CONN_TO, message[1], int(message[2])))
                                message = message[3:] 
                            else:
                                self.__cmds_success.append(False)
                                break
                        else:
                            self.__cmds_success.append(False)
                            break

                    # Start learning
                    elif message[0] == CommunicationProtocol.START_LEARNING:
                        if len(message) > 2:
                            if message[1].isdigit() and message[2].isdigit():
                                self.__cmds_success.append(self.__exec(CommunicationProtocol.START_LEARNING, int(message[1]), int(message[2])))
                                message = message[3:]
                            else:
                                self.__cmds_success.append(False)
                                break
                        else:
                            self.__cmds_success.append(False)
                            break

                    # Stop learning
                    elif message[0] == CommunicationProtocol.STOP_LEARNING:
                        self.__cmds_success.append(self.__exec(CommunicationProtocol.STOP_LEARNING))
                        message = message[1:]
        
                    # Number of samples
                    elif message[0] == CommunicationProtocol.NUM_SAMPLES:
                        if len(message) > 1:
                            if message[1].isdigit():
                                self.__cmds_success.append(self.__exec(CommunicationProtocol.NUM_SAMPLES, int(message[1])))
                                message = message[2:]
                            else:
                                self.__cmds_success.append(False)
                                break
                        else:
                            self.__cmds_success.append(False)
                            break

                    # Ready
                    elif message[0] == CommunicationProtocol.READY:
                        if len(message) > 1:
                            if message[1].isdigit():
                                self.__cmds_success.append(self.__exec(CommunicationProtocol.READY, int(message[1])))
                                message = message[2:]
                            else:
                                self.__cmds_success.append(False)
                                break
                        else:
                            self.__cmds_success.append(False)
                            break
                            
                    # Non Recognized message            
                    else:
                        self.__cmds_success.append(False)
                        break
                
            # Return
            x = self.__cmds_success
            self.__cmds_success = []
            return x

    # Exec callbacks
    def __exec(self,action, *args):
        try:
            self.command_dict[action].execute(*args)
            return True
        except Exception as e:
            logging.info("Error executing callback: " + str(e))
            logging.exception(e)
            return False

    #######################
    #     MSG BUILDERS    # ---->  STATIC METHODS
    #######################

    def build_beat_msg():
        """ 
        Returns:
            A encoded beat message.
        """
        return (CommunicationProtocol.BEAT + "\n").encode("utf-8")

    def build_stop_msg():
        """ 
        Returns:
            A encoded stop message.
        """
        return (CommunicationProtocol.STOP + "\n").encode("utf-8")

    def build_connect_msg(ip, port, broadcast):
        """
        Args:
            ip: The ip address of the node that tries to connect.
            port: The port of the node that tries to connect.
            broadcast: Whether or not to broadcast the message.

        Returns:
            A encoded connect message.
        """
        return (CommunicationProtocol.CONN + " " + ip + " " + str(port) + " " + str(broadcast) + "\n").encode("utf-8")

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
        return (CommunicationProtocol.START_LEARNING + " " + str(rounds) + " " + str(epochs) + "\n").encode("utf-8")

    def build_stop_learning_msg():
        """
        Returns:
            A encoded stop learning message.
        """
        return (CommunicationProtocol.STOP_LEARNING + "\n").encode("utf-8")

    def build_num_samples_msg(num):
        """
        Args:
            num: The number of samples to train.

        Returns:
            A encoded number of samples message.
        """
        return (CommunicationProtocol.NUM_SAMPLES + " " + str(num) + "\n").encode("utf-8")

    def build_ready_msg(round):
        """
        Args:
            round: The last round finished.

        Returns:
            A encoded ready message.
        """
        return (CommunicationProtocol.READY + " " + str(round) + "\n").encode("utf-8")

    def build_params_msg(data):
        """
        Args:
            data: The model parameters to send (encoded).

        Returns:
            A list of fragments messages of the params.
        """
        # Encoding Headers and ending
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        end = CommunicationProtocol.PARAMS_CLOSE.encode("utf-8")

        # Spliting data
        size = BUFFER_SIZE - len(header)
        data_msgs = []
        for i in range(0, len(data), size):
            data_msgs.append(header + (data[i:i+size]))

        # Adding closing message
        if len(data_msgs[-1]) + len(end) <= BUFFER_SIZE:
            data_msgs[-1] += end
            data_msgs[-1] += b'\0' * (BUFFER_SIZE - len(data_msgs[-1])) # padding to avoid message fragmentation
        else:
            data_msgs.append(header + end)

        return data_msgs
