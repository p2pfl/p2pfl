import socket
import threading
import logging
import sys

from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.encrypter import AESCipher, RSACipher
from p2pfl.gossiper import Gossiper
from p2pfl.settings import Settings
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater
from p2pfl.utils.observer import Events, Observer

class BaseNode(threading.Thread, Observer):
    """
    This class represents a base node in the network (without **FL**). It is a thread, so it's going to process all messages in a background thread using the CommunicationProtocol.

    Args:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If the node is in simulation mode or not.

    Attributes:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If the node is in simulation mode or not.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(self, host="127.0.0.1", port=None, simulation=True):
        threading.Thread.__init__(self)
        self.__terminate_flag = threading.Event()
        self.host = socket.gethostbyname(host)
        self.port = port
        self.simulation = simulation

        # Setting Up Node Socket (listening)
        self.__node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP Socket
        if port==None:
            self.__node_socket.bind((host, 0)) # gets a random free port
            self.port = self.__node_socket.getsockname()[1]
        else:
            self.__node_socket.bind((host, port))
        self.__node_socket.listen(50) # no more than 50 connections at queue
        self.name = "node-" + self.get_name()
        
        # neightbors
        self.__neightbors = [] # private to avoid concurrency issues
        self.__nei_lock = threading.Lock()

        # Logging
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        # Heartbeater and Gossiper
        self.gossiper = None
        self.heartbeater = None
        
    def get_addr(self):
        """
        Returns:
            tuple: The address of the node.    
        """
        return self.host,self.port

    def get_name(self):
        """
        Returns:
            str: The name of the node.
        """
        return str(self.get_addr()[0]) + ":" + str(self.get_addr()[1]) 

    #######################
    #   Node Management   #
    #######################
    
    # Start the main loop in a new thread
    def start(self):
        """
        Starts the node. It will listen for new connections and process them. Heartbeater will be started too.
        """
        super().start()
        # Heartbeater and Gossiper
        self.heartbeater = Heartbeater(self.get_name())
        self.gossiper = Gossiper(self.get_name(),self.__neightbors) # only reads neightbors
        self.heartbeater.add_observer(self)
        self.gossiper.add_observer(self)
        self.heartbeater.start()
        self.gossiper.start()
    
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

    def is_stopped(self):
        return self.__terminate_flag.is_set()

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
        logging.info('Nodo listening at {}:{}'.format(self.host, self.port))
        while not self.__terminate_flag.is_set(): 
            try:
                (ns, addr) = self.__node_socket.accept()
                msg = ns.recv(Settings.BLOCK_SIZE)
                
                # Process new connection
                if msg:
                    msg = msg.decode("UTF-8")
                    callback = lambda h,p,fu,fc: self.__process_new_connection(ns, h, p, fu, fc)
                    if not CommunicationProtocol.process_connection(msg,callback):
                        logging.debug('({}) Conexión rechazada con {}:{}'.format(self.get_name(),addr,msg))
                        ns.close()         
                        
            except Exception as e:
                logging.exception(e)

        # Stop Heartbeater and Gossiper
        self.heartbeater.stop()
        self.gossiper.stop()
        # Stop Node
        logging.info('Bajando el nodo, dejando de escuchar en {} {} y desconectándose de {} nodos'.format(self.host, self.port, len(self.__neightbors))) 
        # Al realizar esta copia evitamor errores de concurrencia al recorrer la misma, puesto que sabemos que se van a eliminar los nodos de la misma
        nei_copy_list = self.__neightbors.copy()
        for n in nei_copy_list:
            n.stop()
        self.__node_socket.close()


    def __process_new_connection(self, node_socket, h, p, full, force):
        try:
            # Check if connection with the node already exist
            self.__nei_lock.acquire()
            if self.get_neighbor(h,p,thread_safe=False) == None:

                # Check if ip and port are correct
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                s.settimeout(2) #########################################################################################################################
                result = s.connect_ex((h,p)) 
                s.close()

                aes_cipher = None
                if not self.simulation:
                    # Encryption (asymmetric)
                    rsa = RSACipher()
                    node_socket.sendall(rsa.serialize_key())
                    rsa.load_pair_public_key(node_socket.recv(len(rsa.serialize_key())))

                    # Encryption (symmetric)
                    aes_cipher = AESCipher()
                    node_socket.sendall(aes_cipher.get_key())

                # Add neightboor
                if result == 0:
                    logging.info('{} Conexión aceptada con {}:{}'.format(self.get_name(),h,p))
                    nc = NodeConnection(self.get_name(),node_socket,(h,p),aes_cipher)
                    nc.add_observer(self)
                    self.__neightbors.append(nc)
                    nc.start(force=force)
                    
                    if full:
                        self.broadcast(CommunicationProtocol.build_connect_to_msg(h,p),exc=[nc],thread_safe=False)
            else:
                node_socket.close()
            
            self.__nei_lock.release()
            
        except Exception as e:
            logging.exception(e)
            node_socket.close()
            self.__nei_lock.release()
            self.rm_neighbor(nc)
            raise e
   

    #############################
    #  Neighborhood management  #
    #############################

    def update(self,event,obj):
        """
        Observer update method. Used to handle events that can occur in the agregator or neightbors.
        
        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated. 
        """
        if event == Events.END_CONNECTION:
            self.rm_neighbor(obj)

        elif event == Events.NODE_CONNECTED_EVENT:
            n = obj[0]
            n.send(CommunicationProtocol.build_beat_msg(self.get_name())) # todos los mensajes van a ser necesarior

        elif event == Events.CONN_TO:
            self.connect_to(obj[0], obj[1], full=False)

        elif event == Events.SEND_BEAT_EVENT:
            self.broadcast(CommunicationProtocol.build_beat_msg(self.get_name())) # todos los mensajes van a ser necesarior

        elif event == Events.GOSSIP_BROADCAST_EVENT:
            self.broadcast(obj[0],exc=obj[1]) 
            
        elif event == Events.PROCESSED_MESSAGES_EVENT:

            node, msgs = obj
            #print("Processed {} messages ({})".format(len(msgs),msgs))

            # Comunicate to nodes the new messages processed
            for nc in self.__neightbors:
                nc.add_processed_messages(list(msgs.keys()))

            # Gossip the new messages
            self.gossiper.add_messages(list(msgs.values()),node)

        elif event == Events.BEAT_RECEIVED_EVENT:
            self.heartbeater.add_node(obj)

            
    def get_neighbor(self, h, p, thread_safe=True):
        """
        Get a NodeConnection from the neightbors list.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.

        Returns:
            NodeConnection: The NodeConnection of the node.
        """
        if thread_safe:
            self.__nei_lock.acquire()

        return_node = None    
        for n in self.__neightbors:
            if n.get_addr() == (h,p):
                return_node = n
                break
            
        if thread_safe:
            self.__nei_lock.release()
        
        return return_node

    def get_neighbors(self):
        """
        Returns:
            list: The neightbors of the node.
        """
        self.__nei_lock.acquire()
        n = self.__neightbors.copy()
        self.__nei_lock.release()
        return n

    def rm_neighbor(self,n):
        """
        Removes a neightboor from the neightbors list.

        Args:
            n (NodeConnection): The neightboor to be removed.
        """
        self.__nei_lock.acquire()
        try:
            self.__neightbors.remove(n)
            n.stop()
        except:
            pass
        self.__nei_lock.release()


    def connect_to(self, h, p, full=False, force=False):
        """"
        Connects to a node.
        
        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.

        Returns:
            node: The node that has been connected to.
        """
        if full:
            full = "1"
        else:
            full = "0"
        
        if force:
            force = "1"
        else:
            force = "0"

        try:
            # Check if connection with the node already exist
            h = socket.gethostbyname(h)
            self.__nei_lock.acquire()
            if self.get_neighbor(h,p,thread_safe=False) == None:
            
                # Send connection request
                msg=CommunicationProtocol.build_connect_msg(self.host,self.port,full,force)
                s = self.__send(h,p,msg,persist=True)

                aes_cipher = None
                if not self.simulation:
                    # Encryption (asymetric)
                    rsa = RSACipher()
                    rsa.load_pair_public_key(s.recv(len(rsa.serialize_key())))
                    s.sendall(rsa.serialize_key())
                    # Encryption (symetric)
                    aes_cipher = AESCipher(key=s.recv(AESCipher.key_len()))

                # Add socket to neightbors
                logging.info("{} Connected to {}:{}".format(self.get_name(),h,p))
                nc = NodeConnection(self.get_name(),s,(h,p),aes_cipher)

                nc.add_observer(self)
                self.__neightbors.append(nc)
                nc.start(force=force)
                self.__nei_lock.release()
                return nc
        
            else:
                logging.info("{} Already connected to {}:{}".format(self.get_name(),h,p))
                self.__nei_lock.release()
                return None
    
        except Exception as e:
            logging.info("{} Can't connect to the node {}:{}".format(self.get_name(),h,p))
            logging.exception(e)
            try:
                self.__nei_lock.release()
            except:
                pass
            return None


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

    # A PARTIR DE AHORA, TODOS LOS MENSAJES VAN A SER NECESARIOS
    def broadcast(self, msg, exc=[], thread_safe=True):
        """
        Broadcasts a message to all the neightbors.

        Args:
            msg (str): The message to be broadcasted.
            ttl (int): The time to live of the message.
            exc (list): The neightbors to be excluded.

        Returns:
            bool: If True, the message has been sent.
        """

        # ESTO DE RETURN TRUE YO LO SACABA

        if thread_safe:
            self.__nei_lock.acquire()

        sended=True 
        for n in self.__neightbors: # to avoid concurrent modification
            if not (n in exc):
                sended = sended and n.send(msg)

        if thread_safe:
            self.__nei_lock.release()
        
        return sended

