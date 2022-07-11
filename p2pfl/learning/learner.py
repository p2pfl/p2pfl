
################################
#    NodeLearning Interface    #  -->  Template Pattern
################################

class NodeLearner:
    """
    Template to implement learning processes, iincluding metric monitoring during training.
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

    def encode_parameters(self, params=None, contributors=None, weight=None):
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

    def decode_parameters(self,data): 
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
        Evaluate the model. With a given parameters.
        """
        pass

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        """
        Log the validation metrics. It also can be used to log the othe node metrics.
        """
        pass

    def get_num_samples(self): 
        """
        Get the number of samples of the model.

        Returns:
            The number of samples of the model.
        """
        pass

    def init(self): 
        """
        Init the learner.
        """
        pass

    def close(self): 
        """
        Close the learner.
        """
        pass

    def finalize_round(self):
        """
        Determine the end of the round.
        """
        pass
