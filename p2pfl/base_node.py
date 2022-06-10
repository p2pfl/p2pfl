import socket
import threading
import logging
import sys
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater

class BaseNode(threading.Thread):
    """
    This class represents a base node in the network (without **FL**). It is a thread, so it's going to process all messages in a background thread using the CommunicationProtocol.

    Args:
        host (str): The host of the node.
        port (int): The port of the node.

    Attributes:
        host (str): The host of the node.
        port (int): The port of the node.
        node_socket (socket): The socket of the node.
        neightboors (list): The neightboors of the node.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(self, host="127.0.0.1", port=0):
        threading.Thread.__init__(self)
        self.__terminate_flag = threading.Event()
        self.host = host
        self.port = port

        # Setting Up Node Socket (listening)
        self.node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP Socket
        self.node_socket.bind((host, port))
        self.node_socket.listen(50) # no more than 50 connections at queue
        if port==0:
            self.port = self.node_socket.getsockname()[1]
        self.name = "node-" + self.get_addr()[0] + ":" + str(self.get_addr()[1])
        
        # Neightboors
        self.neightboors = []

        # Logging
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        # Heartbeater
        self.heartbeater = Heartbeater(self)
        self.heartbeater.start()

        
    def get_addr(self):
        """
        Returns:
            tuple: The address of the node.    
        """
        return self.host,self.port

    #######################
    #   Node Management   #
    #######################
    
    # Start the main loop in a new thread
    def start(self):
        """
        Starts the node. It will listen for new connections and process them.
        """
        super().start()

    
    def stop(self): 
        """
        Stops the node.
        """
        self.__terminate_flag.set()
        # Enviamos mensaje al loop para evitar la espera del recv
        try:
            self.__send(self.host,self.port,b"")
        except:
            pass

    def __send(self, h, p, data, persist=False): 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        s.sendall(data)
        if persist:
            return s
        else:
            s.close()
            return None

    ########################
    #   Main Thread Loop   #
    ########################

    # Main Thread of node.
    #   Its listening for new nodes to be added
    def run(self):
        """
        The main loop of the node. It will listen for new connections and process them.
        """
        logging.info('Nodo a la escucha en {} {}'.format(self.host, self.port))
        while not self.__terminate_flag.is_set(): 
            try:
                (node_socket, addr) = self.node_socket.accept()
                msg = node_socket.recv(BUFFER_SIZE)
                
                # Process new connection
                if msg:
                    msg = msg.decode("UTF-8")
                    callback = lambda h,p,b: self.__process_new_connection(node_socket, h, p, b)
                    if not CommunicationProtocol.process_connection(msg,callback):
                        logging.debug('({}) Conexión rechazada con {}:{}'.format(self.get_addr(),addr,msg))
                        node_socket.close()         
                        
            except Exception as e:
                logging.exception(e)

        #Stop Node
        logging.info('Bajando el nodo, dejando de escuchar en {} {} y desconectándose de {} nodos'.format(self.host, self.port, len(self.neightboors))) 
        # Al realizar esta copia evitamor errores de concurrencia al recorrer la misma, puesto que sabemos que se van a eliminar los nodos de la misma
        nei_copy_list = self.neightboors.copy()
        for n in nei_copy_list:
            n.stop()
        self.heartbeater.stop()
        self.node_socket.close()


    def __process_new_connection(self, node_socket, h, p, broadcast):
        try:
            # Check if connection with the node already exist
            if self.get_neighbor(h,p) == None:

                # Check if ip and port are correct
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                s.settimeout(2)
                result = s.connect_ex((h,p)) 
                s.close()
        
                # Add neightboor
                if result == 0:
                    logging.info('({}) Conexión aceptada con {}'.format(self.get_addr(),(h,p)))
                    nc = NodeConnection(self,node_socket,(h,p))
                    nc.add_observer(self)
                    nc.start()
                    self.add_neighbor(nc)
                    
                    if broadcast:
                        self.broadcast(CommunicationProtocol.build_connect_to_msg(h,p),exc=[nc])
            else:
                node_socket.close()

        except Exception as e:
            logging.exception(e)
            node_socket.close()
            self.rm_neighbor(nc)
            raise e
   

    #############################
    #  Neighborhood management  #
    #############################

    def get_neighbor(self, h, p):
        """
        Get a NodeConnection from the neightboors list.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.

        Returns:
            NodeConnection: The NodeConnection of the node.
        """
        for n in self.neightboors:
            if n.get_addr() == (h,p):
                return n
        return None

    def get_neighbors(self):
        """
        Returns:
            list: The neightboors of the node.
        """
        return self.neightboors

    def add_neighbor(self, n):
        """
        Adds a neightboor to the neightboors list.
            
        Args:
            n (NodeConnection): The neightboor to be added.
        """
        self.neightboors.append(n)

    def rm_neighbor(self,n):
        """
        Removes a neightboor from the neightboors list.

        Args:
            n (NodeConnection): The neightboor to be removed.
        """
        try:
            self.neightboors.remove(n)
        except:
            pass

    def connect_to(self, h, p, full=True):
        """"
        Connects to a node.
        
        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
        """
        if full:
            full = "1"
        else:
            full = "0"
            
        # Check if connection with the node already exist
        if self.get_neighbor(h,p) == None:

            # Send connection request
            msg=CommunicationProtocol.build_connect_msg(self.host,self.port,full)
            s = self.__send(h,p,msg,persist=True)
            
            # Add socket to neightboors
            nc = NodeConnection(self,s,(h,p))
            nc.add_observer(self)
            nc.start()
            self.add_neighbor(nc)
        
        else:
            logging.info('El nodo ya se encuentra conectado con {}:{}'.format(h,p))

    def disconnect_from(self, h, p):
        """
        Disconnects from a node.
        
        Args:
            h (str): The host of the node.
            p (int): The port of the node.
        """
        self.get_neighbor(h,p).stop()
      
    ##########################
    #     Msg management     #
    ##########################

    # FULL_CONNECTED -> 3th iteration |> TTL 
    def broadcast(self, msg, ttl=1, exc=[], is_necesary=True):
        """
        Broadcasts a message to all the neightboors.

        Args:
            msg (str): The message to be broadcasted.
            ttl (int): The time to live of the message.
            exc (list): The neightboors to be excluded.
            is_necesary (bool): If False, the message will be sent only if its posible.
        """
        sended=True 
        for n in self.neightboors:
            if not (n in exc):
                sended = sended and n.send(msg, is_necesary)
        return sended

