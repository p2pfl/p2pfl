class BaseNodeState:
    """
    Class to store the state of a base node. Empty in this case (in the future can be added the neightboors).
    """

    def __init__(self):
        self.status = "Idle"
        super().__init__()


class NodeState(BaseNodeState):
    """
    Class to store the main state of a learning node.
    """

    def __init__(self):
        super().__init__()
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """
        Set the experiment name.
        """
        self.status = "Learning"
        self.actual_exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = 0

    def increase_round(self) -> None:
        """
        Increase the round number.
        """
        self.round += 1

    def clear(self) -> None:
        """
        Clear the state.
        """
        self.status = "Idle"
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None
