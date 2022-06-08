##################################
#    Generic Observable class    #
##################################
class Events():
    END_CONNECTION = "END_CONNECTION"
    NODE_READY_EVENT = "NODE_READY_EVENT"

##################################
#    Generic Observable class    #
##################################

class Observable():
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify(self, nc, event) -> None:
        [o.update(nc, event) for o in self.observers]

##################################
#    Generic Observer class    #
##################################

class Observer():
    def update(self, nc, event): pass