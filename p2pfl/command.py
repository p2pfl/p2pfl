import threading

#################
#    Command    #
#################

class Command:
    def __init__(self, parent_node, node_connection):
        self.parent_node = parent_node
        self.node_connection = node_connection

    def execute(self,*args): pass

########################
#       Commands       #
########################

class Beat_cmd(Command):
    def execute(self):
        pass #logging.debug("Beat {}".format(self.get_addr()))

class Stop_cmd(Command):
    def execute(self):
        self.node_connection.stop(local=True)

class Conn_to_cmd(Command):
    def execute(self,h,p):
        self.parent_node.connect_to(h, p, full=False)

class Start_learning_cmd(Command):
    def execute(self, rounds, epochs):
        # Thread process to avoid blocking the message receiving
        learning_thread = threading.Thread(target=self.parent_node.start_learning,args=(rounds,epochs))
        learning_thread.start()

class Stop_learning_cmd(Command):
    def execute(self):
        self.parent_node.stop_learning()

class Num_samples_cmd(Command):
    def execute(self,num):
        self.node_connection.set_num_samples(num)

class Params_cmd(Command):
    def execute(self,msg,done):
        self.node_connection.add_param_segment(msg)
        if done:
            params = self.node_connection.get_params()
            self.node_connection.clear_buffer()
            self.parent_node.add_model(params,self.node_connection.num_samples)
        