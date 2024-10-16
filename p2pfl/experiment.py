from datetime import datetime

class Experiment:
    def __init__(self, exp_name: str, total_rounds: int):
        self.exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = 0

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the round is not initialized.
        """
        if self.round is None:
            raise ValueError("Round not initialized")
        self.round += 1

    def self(self, param_name, param_val=None):
        if param_val is None:
            return getattr(self, param_name)
        else:
            setattr(self, param_name, param_val)

    def __str__(self):
        return (f"Experiment(exp_name={self.exp_name}, total_rounds={self.total_rounds}, "
                f"round={self.round})")