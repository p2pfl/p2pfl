"""
Module that implements commands of the command pattern.
"""

#################
#    Command    #
#################

class Command:
    """
    Class that represents a command.

    Args:
        node_connection: The node connection that is going to execute the command.
    """
    def __init__(self, node_connection):
        self.node_connection = node_connection

    def execute(self,*args): pass

########################
#       Commands       #
########################

class Beat_cmd(Command):
    """
    Command that should be executed as a response to a **beat** message.
    """
    def execute(self, node1):
        self.node_connection.notify_heartbeat(node1)

class Stop_cmd(Command):
    """
    Command that is should be executed as a response to a **stop** message.
    """
    def execute(self):
        self.node_connection.stop(local=True)

class Conn_to_cmd(Command):
    """
    Command that should be executed as a response to a **conn_to** message.
    """
    def execute(self,h,p):
        self.node_connection.notify_conn_to(h,p)

class Start_learning_cmd(Command):
    """
    Command that should be executed as a response to a **start_learning** message.
    """
    def execute(self, rounds, epochs):
        self.node_connection.notify_start_learning(rounds, epochs)

class Stop_learning_cmd(Command):
    """
    Command that should be executed as a response to a **stop_learning** message.
    """
    def execute(self):
        self.node_connection.notify_stop_learning()

class Params_cmd(Command):
    """
    Command that should be executed as a response to a **params** message.
    """
    def execute(self,msg,done):
        self.node_connection.add_param_segment(msg)
        if done:
            params = self.node_connection.get_params()
            self.node_connection.clear_buffer()
            self.node_connection.notify_params(params)

class Models_Ready_cmd(Command):
    """
    Command that should be executed as a response to a **ready** message.
    """
    def execute(self,round):
        self.node_connection.set_model_ready_status(round)

class Metrics_cmd(Command):
    """
    Command that should be executed as a response to a **metrics** message.
    """
    def execute(self, node, round, loss, metric):
        self.node_connection.notify_metrics(node,round,loss,metric)


class Vote_train_set_cmd(Command):
    """
    Command that should be executed as a response to a **vote** message.
    """
    def execute(self, node, votes):
        self.node_connection.notify_train_set_votes(node, votes)

class Models_agregated_cmd(Command):
    """
    Command that should be executed as a response to a **models_agregated** message.
    """
    def execute(self,node_list):
        self.node_connection.add_models_agregated(node_list)

class Model_initialized_cmd(Command):
    """
    Command that should be executed as a response to a **model_initialized** message.
    """
    def execute(self):
        self.node_connection.set_model_initialized(True)