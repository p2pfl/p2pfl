import ray
from datetime import datetime

class Experiment:
    def __init__(self, exp_name: str, total_rounds: int, round: int, timestamp: datetime):
        self.exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = round
        self.timestamp = timestamp

    def __str__(self):
        return (f"LocalExperiment(exp_name={self.exp_name}, total_rounds={self.total_rounds}, "
                f"round={self.round}, timestamp={self.timestamp})")

@ray.remote
class ExperimentActor:
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

    def clear(self) -> None:
        """Clear the experiment state."""
        self.exp_name = None
        self.round = None
        self.total_rounds = None

    def self(self, param_name, param_val=None):
        if param_val is None:
            return getattr(self, param_name)
        else:
            setattr(self, param_name, param_val)

    def get_experiment(self):
        """Return a Experiment instance with the current state and timestamp."""
        state = {
            "exp_name": self.exp_name,
            "total_rounds": self.total_rounds,
            "round": self.round
        }
        timestamp = datetime.now()
        return Experiment(
            exp_name=state["exp_name"],
            total_rounds=state["total_rounds"],
            round=state["round"],
            timestamp=timestamp
        )

    def __str__(self) -> str:
        """String representation of the experiment state."""
        return f"Experiment(exp_name={self.exp_name}, round={self.round}, total_rounds={self.total_rounds})"