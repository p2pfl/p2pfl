"""
Module that implements the command pattern.
"""

import threading

#################
#    Command    #
#################

#
# REVISAR BIEN ESTA CONEXIÓN, SEGURAMENTE PUEDA SER MEJORADA USANDO EL PATRON OBSERVER
#

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
    def execute(self):
        pass #logging.debug("Beat {}".format(self.get_addr()))

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

class Num_samples_cmd(Command):
    """
    Command that should be executed as a response to a **num_samples** message.
    """
    def execute(self,num):
        self.node_connection.set_num_samples(num)

class Ready_cmd(Command):
    """
    Command that should be executed as a response to a **ready** message.
    """
    def execute(self,round):
        self.node_connection.set_ready_status(round)

class Params_cmd(Command):
    """
    Command that should be executed as a response to a **params** message.
    """
    def execute(self,msg,done):
        self.node_connection.add_param_segment(msg)
        if done:
            self.node_connection.tmp = self.node_connection.tmp + 1
            params = self.node_connection.get_params()
            self.node_connection.clear_buffer()
            self.node_connection.notify_params(params)

        