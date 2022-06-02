
################################
#    NodeLearning Interface    #  -->  Template Pattern
################################

class NodeLearner:

    def set_model(self, model): pass

    def set_data(self, data): pass

    def encode_parameters(self): pass

    def decode_parameters(self,data): pass

    def set_parameters(self, params): pass

    def get_parameters(self): pass

    def set_epochs(self, epochs): pass

    def fit(self): pass

    def interrupt_fit(self): pass

    def evaluate(self): pass

    def predict(self): pass

    def get_num_samples(self): pass

