# learning/callbacks/tensorflow/scaffold_callback.py

from learning.callbacks.decorators import register
from tensorflow.keras.callbacks import Callback
import numpy as np
import copy

@register(aggregator_type='scaffold')
class SCAFFOLDCallback(Callback):
    """
    Callback for SCAFFOLD operations to use with TensorFlow Keras.
    
    At the beginning of the training, the callback stores the global model and the initial learning rate.
    After optimization steps, it applies control variate adjustments.
    """
    
    def __init__(self, aggregator):
        """Initialize the callback."""
        super().__init__()
        self.c_i: Optional[List[np.ndarray]] = None
        self.c: Optional[List[np.ndarray]] = None
        self.initial_model_weights: Optional[List[np.ndarray]] = None
        self.saved_lr: Optional[float] = None
        self.K: int = 0
        self.aggregator = aggregator

    def on_train_begin(self, logs=None):
        """
        Store the global control variate and reset local control variates.
        
        Args:
            logs: Optional logging dictionary.
        """
        if self.c_i is None:
            self.c_i = [np.zeros_like(weight) for weight in self.model.get_weights()]

        if self.K == 0:
            self._get_global_c()

        self.initial_model_weights = copy.deepcopy(self.model.get_weights())
        self.K = 0

    def on_batch_end(self, batch, logs=None):
        """
        Apply control variate adjustments after each batch.
        
        Args:
            batch: Batch number.
            logs: Optional logging dictionary.
        """
        if not hasattr(self.model.optimizer, 'lr'):
            raise AttributeError("Optimizer must have 'lr' attribute.")
        self.saved_lr = self.model.optimizer.lr.numpy()

        eta_l = self.saved_lr
        weights = self.model.get_weights()
        adjusted_weights = []
        for idx, (weight, c_i_param, c_param) in enumerate(zip(weights, self.c_i, self.c)):
            adjusted_weight = weight + eta_l * c_i_param - eta_l * c_param
            adjusted_weights.append(adjusted_weight)
        self.model.set_weights(adjusted_weights)
        self.K += 1

    def on_train_end(self, logs=None):
        """
        Update control variates and store deltas for aggregation at the end of training.
        
        Args:
            logs: Optional logging dictionary.
        """
        if self.K == 0:
            raise ValueError("Local steps K must be greater than 0.")
        if self.c_i is None:
            raise ValueError("Local control variate c_i is not initialized.")
        if self.initial_model_weights is None:
            raise ValueError("Initial model weights are not stored.")

        y_i = self.model.get_weights()
        x_g = self.initial_model_weights
        previous_c_i = [c.copy() for c in self.c_i]

        for idx, (c_i, x, y) in enumerate(zip(self.c_i, x_g, y_i)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx] = c_i - adjustment

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y, x in zip(y_i, x_g)]
        delta_c_i = [c_new - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]

        # Convert to NumPy arrays for transmission
        delta_y_i_np = [dyi for dyi in delta_y_i]
        delta_c_i_np = [dci for dci in delta_c_i]

        self.model.add_info('delta_y_i', delta_y_i_np)
        self.model.add_info('delta_c_i', delta_c_i_np)

    def _get_global_c(self):
        """
        Retrieve the global control variate from the aggregator.
        """
        c_np_list = self.aggregator.get_global_control_variate()
        self.c = c_np_list
