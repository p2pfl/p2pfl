##################################
#    Generic Observable class    #
##################################
class Events():
    END_CONNECTION = "END_CONNECTION"
    NODE_READY_EVENT = "NODE_READY_EVENT"
    AGREGATION_FINISHED = "AGREGATION_FINISHED"

##################################
#    Generic Observable class    #
##################################

class Observable():
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
    def update(self, event, obj): pass