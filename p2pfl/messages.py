class NodeMessages:
    """
    Class that contains the messages exchanged between nodes.
    """

    BEAT: str = "beat"


class LearningNodeMessages(NodeMessages):
    """
    Class that contains the messages exchanged between learning nodes.
    """

    START_LEARNING: str = "start_learning"
    STOP_LEARNING: str = "stop_learning"
    MODEL_INITIALIZED: str = "model_initialized"
    MODELS_AGGREGATED: str = "models_aggregated"
    MODELS_READY: str = "models_ready"
    VOTE_TRAIN_SET: str = "vote_train_set"
    METRICS: str = "metrics"
