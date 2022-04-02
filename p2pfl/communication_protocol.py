
class CommunicationProtocol:

    BEAT     = "BEAT"
    STOP     = "STOP"
    CONN     = "CONNECT"
    CONN_TO  = "CONNECT_TO"

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
    #
    def process_message(self, message):
        message = message.split()
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
            
        else:
            return False


    # Exec callbacks
    def __exec(self,action, *args):
        try:
            self.callback_dict[action](*args)
            return True
        except:
            return False
