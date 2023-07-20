#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

################################
#    NodeLearning Interface    #  -->  Template Pattern
################################


class NodeLearner:
    """
    Template to implement learning processes, including metric monitoring during training.
    """

    def set_model(self, model):
        """
        Set the model of the learner. (not wheights)

        Args:
            model: The model of the learner.

        Raises:
            ModelNotMatchingError: If the model is not matching the learner.
        """
        pass

    def set_data(self, data):
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.
        """
        pass

    def encode_parameters(self, params=None):
        """
        Encode the parameters of the model. (binary)
        If params are not provided, self parameters are encoded.

        Args:
            params: The parameters of the model. (non-binary)
            contributors: The contributors of the model.
            weight: The weight of the model.

        Returns:
            The encoded parameters of the model (params, contributors, weight).
        """
        pass

    def decode_parameters(self, data):
        """
        Decode the parameters of the model. (binary)

        Args:
            data: The encoded parameters of the model.

        Returns:
            The decoded parameters of the model. (params, contributors, weight)

        Raises:
            DecodingParamsError: If the decoding of the parameters fails.
        """
        pass

    def check_parameters(self, params):
        """
        Check if the parameters are compatible with the model.

        Args:
            params: The parameters to check. (non-binary)

        Returns:
            True if the parameters are compatible with the model.
        """
        pass

    def set_parameters(self, params):
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model. (non-binary)

        Raises:
            ModelNotMatchingError: If the model is not matching the learner.
        """
        pass

    def get_parameters(self):
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model. (non-binary)
        """
        pass

    def set_epochs(self, epochs):
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.
        """
        pass

    def fit(self):
        """
        Fit the model.
        """
        pass

    def interrupt_fit(self):
        """
        Interrupt the fit process.
        """
        pass

    def evaluate(self):
        """
        Evaluate the model with actual parameters.
        """
        pass

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        """
        Log the validation metrics. It also can be used to log the other node metrics.
        """
        pass

    def get_num_samples(self):
        """
        Get the number of samples of the model.

        Returns:
            The number of samples of the model.
        """
        pass

    def finalize_round(self):
        """
        Determine the end of the round.
        """
        pass
