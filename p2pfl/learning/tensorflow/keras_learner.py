#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Keras learner."""

# import logging
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# from p2pfl.learning.exceptions import ModelNotMatchingError
# from p2pfl.learning.learner import NodeLearner
# from p2pfl.learning.p2pfl_model import P2PFLModel
# from p2pfl.learning.pytorch.lightning_logger import FederatedLogger
# from p2pfl.management.logger import logger

# print("AL IGUAL QUE CON PYTORCH, FACILITAR LA INICIALIZACI'ON DE MODELOS")


# ###########################
# #    LightningDataset     #
# ###########################


# class KerasDataset:
#     pass


# #########################
# #    LightningModel     #
# #########################


# class KerasModel(P2PFLModel):
#     """
#     P2PFL model abstraction for PyTorch Lightning.

#     Args:
#         model: The model to encapsulate.

#     """

#     def __init__(
#         self, model: tf.keras.Model, input_shape: Tuple[int, ...], aditional_info: Optional[Dict[str, str]] = None
#     ) -> None:
#         """Initialize the model."""
#         self.model = model
#         self.additional_info = aditional_info
#         self.contributors: Dict[str, int] = {}
#         if self.additional_info is not None:
#             self.additional_info = aditional_info

#         # Force initialization by calling the model on a dummy input
#         self.model(keras.Input(shape=input_shape))

#     def get_parameters(self) -> List[np.ndarray]:
#         """
#         Get the parameters of the model.

#         Returns:
#             The parameters of the model

#         """
#         return self.model.get_weights()

#     def set_parameters(self, params: List[np.ndarray]) -> None:
#         """
#         Set the parameters of the model.

#         Args:
#             params: The parameters of the model.

#         Raises:
#             ModelNotMatchingError: If parameters don't match the model.

#         """
#         try:
#             self.model.set_weights(params)
#         except Exception as e:
#             raise ModelNotMatchingError(f"Not matching models: {e}") from e


# class KerasLearner(NodeLearner):
#     """
#     Learner with Tensorflow Keras.

#     Args:
#         model: The model of the learner.
#         data: The data of the learner.
#         self_addr: The address of the learner.
#         epochs: The number of epochs of the model.

#     """

#     def __init__(self, model: KerasModel, data: KerasDataset, self_addr: str, epochs: int) -> None:
#         """Initialize the learner."""
#         self.model = model
#         self.data = data
#         self.__self_addr = self_addr
#         self.epochs = epochs

#         # Start logging
#         print("HACER EL LOGGER")

#     def set_model(self, model: tf.keras.Model) -> None:
#         """
#         Set the model of the learner.

#         Args:
#             model: The model of the learner.

#         """
#         self.model = model

#     def set_data(self, data: KerasDataset) -> None:
#         """
#         Set the data of the learner.

#         Args:
#             data: The data of the learner.

#         """
#         self.data = data
#         print("esto estaba hardcodeado a tuplas")

#     def get_num_samples(self) -> Tuple[int, int]:
#         """
#         Get the number of samples in the train and test datasets.

#         Args:
#             data: The data of the learner.

#         .. todo:: Use it to obtain a more accurate metric aggretation.

#         """
#         raise NotImplementedError

#     ####
#     # Training
#     ####

#     def set_epochs(self, epochs: int) -> None:
#         """
#         Set the number of epochs.

#         Args:
#             epochs: The number of epochs.

#         """
#         self.epochs = epochs

#     def fit(self) -> None:
#         """Fit the model."""
#         try:
#             if self.epochs > 0:
#                 raise NotImplementedError
#                 self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#                 self.model.fit(self.data[0], epochs=self.epochs, validation_data=self.data[1], verbose=0)
#         except Exception as e:
#             logger.error(self.__self_addr, f"Error in training with Keras: {e}")

#     def interrupt_fit(self) -> None:
#         """Interrupt the fit."""
#         raise NotImplementedError
#         logger.log_info(self.__self_addr, "Interrupting training.")
#         self.model.stop_training = True

#     def evaluate(self) -> Dict[str, float]:
#         """
#         Evaluate the model with actual parameters.

#         Returns:
#             The evaluation results.

#         """
#         raise NotImplementedError
#         try:
#             if self.epochs > 0:
#                 results = self.model.evaluate(self.data[1], verbose=0)
#                 results_dict = dict(zip(self.model.metrics_names, results))
#                 for k, v in results_dict.items():
#                     logger.log_metric(self.__self_addr, k, v)
#                 return results_dict
#             else:
#                 return {}
#         except Exception as e:
#             logger.error(
#                 self.__self_addr,
#                 f"Evaluation error. Something went wrong with Keras. {e}",
#             )
#             raise e
