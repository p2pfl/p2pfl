from typing import Dict, Any

class LearnerStateDTO:
    """
    DTO class for storing the state of the learner (weights and additional info)
    """
    def __init__(self):
        self.weights = {}
        self.additional_info = {}
    
    def add_weights(self, key: str, value: Any):
        """
        Adds a weight pair (layer name, weights) to the learner state.
        """
        self.weights[key] = value
    
    def add_weights_dict(self, weights: Dict[str, Any]):
        """
        Add a dictionary of weights to the learner state.
        """
        self.weights = weights
        
    def add_info(self, key: str, value: Any):
        """
        Adds additional information to the learner state.
        """
        self.additional_info[key] = value
    
    def get_weights(self) -> Dict[str, Dict[str, Any]]:
        return self.weights
    
    