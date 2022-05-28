import logging
from p2pfl.const import BUFFER_SIZE

###############################
#    CommunicationProtocol    #
###############################
#
# Valid messages: 
#   - BEAT
#   - STOP
#   - CONNECT <ip> <port> <broadcast>
#   - CONNECT_TO <ip> <port>
#   - START_LEARNING <rounds> <epoches>
#   - STOP_LEARNING
#   - NUM_SAMPLES <num> 
#   - PARAMS <data> \PARAMS

class CommunicationProtocol:

    BEAT           = "BEAT"
    STOP           = "STOP"
    CONN           = "CONNECT"
    CONN_TO        = "CONNECT_TO"
    START_LEARNING = "START_LEARNING"
    STOP_LEARNING  = "STOP_LEARNING"
    NUM_SAMPLES    = "NUM_SAMPLES"
    PARAMS         = "PARAMS" 

    ########################
    #    MSG PROCESSING    #
    ########################

    def __init__(self, callback_dict):
        self.callback_dict = callback_dict

        """ DEBUG MSGS
        import random
        self.random = random.randrange(0, 100)
        """
    
    # Check if connection is correct and execute the callback (static)
    def process_connection(message, callback):
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
    

    # Check if the message is correct and execute the callback
    def process_message(self, msg):
        """" DEBUG MSGS
        f = open("logs/communication" + str(self.random) + ".log", "a")
        f.write(str(msg))
        f.write("\n-----------------------------------------------------\n")
        f.close()
        """

        header = CommunicationProtocol.PARAMS.encode("utf-8")
        if msg[0:len(header)] == header:
            end = ("\\" + CommunicationProtocol.PARAMS).encode("utf-8")

            # Check if done
            end_pos = msg.find(end)
            if end_pos != -1:
                return self.__exec(CommunicationProtocol.PARAMS, msg[len(header):end_pos], True)

            return self.__exec(CommunicationProtocol.PARAMS, msg[len(header):], False)

        else:      

            # Try to decode the message
            message = ""
            try:
                message = msg.decode("utf-8")
                message = message.split()
            except:
                return False

            # Check message and exec message
            if len(message) > 0:
                # Beat
                if message[0] == CommunicationProtocol.BEAT:
                    return True

                # Stop
                elif message[0] == CommunicationProtocol.STOP:
                    return self.__exec(message[0])

                # Connect to
                elif message[0] == CommunicationProtocol.CONN_TO:
                    if len(message) > 2:
                        if message[2].isdigit():
                            return self.__exec(CommunicationProtocol.CONN_TO, message[1], int(message[2]))
                        else:
                            return False
                    else:
                        return False

                # Start learning
                elif message[0] == CommunicationProtocol.START_LEARNING:
                    if len(message) > 2:
                        if message[1].isdigit() and message[2].isdigit():
                            return self.__exec(CommunicationProtocol.START_LEARNING, int(message[1]), int(message[2]))
                        else:
                            return False
                    else:
                        return False

                # Stop learning
                elif message[0] == CommunicationProtocol.STOP_LEARNING:
                    return self.__exec(CommunicationProtocol.STOP_LEARNING)
    
                # Number of samples
                elif message[0] == CommunicationProtocol.NUM_SAMPLES:
                    if len(message) > 1:
                        if message[1].isdigit():
                            return self.__exec(CommunicationProtocol.NUM_SAMPLES, int(message[1]))
                        else:
                            return False
                    else:
                        return False
                        
                # Non Recognized message            
                else:
                    return False
                
            # Empty message
            else:
                return False


    # Exec callbacks
    def __exec(self,action, *args):
        try:
            self.callback_dict[action](*args)
            return True
        except Exception as e:
            logging.info("Error executing callback: " + str(e))
            logging.exception(e)
            return False

    #######################
    #     MSG BUILDERS    # ---->  STATIC METHODS
    #######################

    def build_beat_msg():
        return CommunicationProtocol.BEAT.encode("utf-8")

    def build_stop_msg():
        return CommunicationProtocol.STOP.encode("utf-8")

    def build_connect_msg(ip, port, broadcast):
        return (CommunicationProtocol.CONN + " " + ip + " " + str(port) + " " + str(broadcast)).encode("utf-8")

    def build_connect_to_msg(ip, port):
        return (CommunicationProtocol.CONN_TO + " " + ip + " " + str(port)).encode("utf-8")

    def build_start_learning_msg(rounds, epochs):
        return (CommunicationProtocol.START_LEARNING + " " + str(rounds) + " " + str(epochs)).encode("utf-8")

    def build_stop_learning_msg():
        return CommunicationProtocol.STOP_LEARNING.encode("utf-8")

    def build_num_samples_msg(num):
        return (CommunicationProtocol.NUM_SAMPLES + " " + str(num)).encode("utf-8")

    # Revisar si se puede parametrizar para no sobresegmentar el mensaje
    def build_params_msg(data):
        # Encoding Headers and ending
        header = CommunicationProtocol.PARAMS.encode("utf-8")
        end = ("\\" + CommunicationProtocol.PARAMS).encode("utf-8")

        # Spliting data
        size = BUFFER_SIZE - len(header)
        data_msgs = []
        for i in range(0, len(data), size):
            data_msgs.append(header + (data[i:i+size]))

        # Adding closing message
        if len(data_msgs[-1]) + len(end) <= BUFFER_SIZE:
            data_msgs[-1] += end
            data_msgs[-1] += b'\0' * (BUFFER_SIZE - len(data_msgs[-1])) # agregamos padding para evitar que pueda solaparse con otro mensaje
        else:
            data_msgs.append(header + end)

        return data_msgs
