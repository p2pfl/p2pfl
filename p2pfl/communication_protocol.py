
import logging
from p2pfl.const import BUFFER_SIZE

# CONSTRUIR AQUÍ MENSAJES QUE SE ENVIAN
# (CommunicationProtocol.START_LEARNING + " " + str(rounds) + " " + str(epoches)).encode("utf-8") -> CommunicationProtocol.build_start_learning_msg(rounds,epoches)

class CommunicationProtocol:

    BEAT           = "BEAT"
    STOP           = "STOP"
    CONN           = "CONNECT"
    CONN_TO        = "CONNECT_TO"
    START_LEARNING = "START_LEARNING"
    STOP_LEARNING  = "STOP_LEARNING"
    PARAMS         = "PARAMS" 


    # Revisar si se puede parametrizar para no sobresegmentar el mensaje

    def build_data_msgs(data):
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

    # 
    # TENER EN CUENTA QUE LOS PESOS SE VAN A ESTAR COMPARTIENDO PARALELAMENTE -> paralelizar
    # 
    # 


    #initialize the communication protocol
    def __init__(self, callback_dict):
        self.callback_dict = callback_dict

        #import random
        #self.random = random.randrange(0, 100)

    # Check if connection is correct and execute the callback
    #
    # CONNECT <ip> <port> <broadcast>
    #
    def process_connection(self, message, callback):
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
    #
    # BEAT
    # STOP
    # CONNECT_TO <ip> <port>
    # START_LEARNING <rounds> <epoches>
    # STOP_LEARNING
    # PARAMS <data> \PARAMS
    #
    def process_message(self, msg):
        #Debuguear Mensajes
        """"
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

            #logging.debug("Processing message: " + str(message))

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
