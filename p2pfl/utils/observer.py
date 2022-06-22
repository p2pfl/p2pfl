"""
Module that implements the observer pattern.
"""

##################################
#    Generic Observable class    #
##################################

class Events():
    """
    Class that represents the events that can be observed.
    """
    END_CONNECTION = "END_CONNECTION"
    NODE_MODELS_READY_EVENT = "NODE_MODELS_READY_EVENT"
    NODE_METRICS_READY_EVENT = "NODE_MODELS_READY_EVENT"
    AGREGATION_FINISHED = "AGREGATION_FINISHED"
    CONN_TO = "CONN_TO"
    START_LEARNING = "START_LEARNING"
    STOP_LEARNING = "STOP_LEARNING"
    PARAMS_RECEIVED = "PARAMS_RECEIVED"
    METRICS_RECEIVED = "METRICS_RECEIVED"

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