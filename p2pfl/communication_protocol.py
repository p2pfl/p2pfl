
import logging


class CommunicationProtocol:

    BEAT           = "BEAT"
    STOP           = "STOP"
    CONN           = "CONNECT"
    CONN_TO        = "CONNECT_TO"
    START_LEARNING = "START_LEARNING"
    STOP_LEARNING = "STOP_LEARNING"
    PARAMS          = "PARAMS" 


    def build_data_msg(data):
        return CommunicationProtocol.PARAMS + " " + str(data) + " \\" + CommunicationProtocol.PARAMS
    # 
    # 
    # TENER EN CUENTA QUE LOS PESOS SE VAN A ESTAR COMPARTIENDO PARALELAMENTE -> paralelizar
    # 
    # 


    #initialize the communication protocol
    def __init__(self, callback_dict):
        self.callback_dict = callback_dict

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
    # START_LEARNING <rounds>
    # STOP_LEARNING
    # PARAMS <data> \PARAMS
    #
    def process_message(self, msg):
        message = msg.split()
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
                if len(message) > 1:
                    if message[1].isdigit():
                        return self.__exec(CommunicationProtocol.START_LEARNING, int(message[1]))
                    else:
                        return False
                else:
                    return False

            # Stop learning
            elif message[0] == CommunicationProtocol.STOP_LEARNING:
                return self.__exec(CommunicationProtocol.STOP_LEARNING)

            
            # Params PARAMS <info> \PARAMS
            elif message[0] == CommunicationProtocol.PARAMS:
                
                logging.debug("Processing params: " + msg)

                if len(message) > 1:
                    done = message[-1] == "\\" + CommunicationProtocol.PARAMS

                    if done:
                        return self.__exec(CommunicationProtocol.PARAMS, " ".join(message[1:-1]), done)
                    else:
                        return self.__exec(CommunicationProtocol.PARAMS, " ".join(message[1:]), done)
                else:
                    return False

        
            
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
