"""
Module that implements the observer pattern.
"""

##################################
#    Generic Observable class    #
##################################

class Events():
    """
    Class that represents the events that can be observed.

    Attributes:
        EXPLICAR EVENTOS Y SUS ARGUMENTOS CUANDO SE NOTIFICAM
    """
    SEND_BEAT_EVENT                 = "SEND_BEAT_EVENT"
    END_CONNECTION                  = "END_CONNECTION"
    AGREGATION_FINISHED             = "AGREGATION_FINISHED"
    CONN_TO                         = "CONN_TO"
    START_LEARNING                  = "START_LEARNING"
    STOP_LEARNING                   = "STOP_LEARNING"
    PARAMS_RECEIVED                 = "PARAMS_RECEIVED"
    METRICS_RECEIVED                = "METRICS_RECEIVED"
    TRAIN_SET_VOTE_RECEIVED_EVENT   = "TRAIN_SET_VOTE_RECEIVED_EVENT"
    NODE_CONNECTED_EVENT            = "NODE_CONNECTED_EVENT"
    PROCESSED_MESSAGES_EVENT        = "PROCESSED_MESSAGES_EVENT"
    GOSSIP_BROADCAST_EVENT          = "GOSSIP_BROADCAST_EVENT"
    BEAT_RECEIVED_EVENT             = "BEAT_RECEIVED_EVENT"
    NODE_DISCONNECTED               = "NODE_DISCONNECTED"

##################################
#    Generic Observable class    #
##################################

class Observable():
    """
    Class that implements the **Observable** at the observer pattern.
    """
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)
    
    def get_observers(self):
        return self.observers

    def notify(self, event, obj) -> None:
        [o.update(event, obj) for o in self.observers]

##################################
#    Generic Observer class    #
##################################

class Observer():
    """
    Class for the **Observer** at the observer pattern.
    """
    def update(self, event, obj): pass