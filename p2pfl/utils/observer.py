##################################
#    Generic Observable class    #
##################################

class Observable():
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify(self,*args) -> None:
        [o.update(*args) for o in self.observers]

##################################
#    Generic Observer class    #
##################################

class Observer():
    def update(self, *args): pass