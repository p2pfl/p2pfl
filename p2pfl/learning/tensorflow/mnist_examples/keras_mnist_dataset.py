import tensorflow as tf
from sklearn.model_selection import train_test_split

class MnistFederatedDM:
    """
    Federated Data Module for MNIST using TensorFlow/Keras. This class manages data partitioning for federated learning.
    
    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        val_percent: The percentage of the validation set.
    """

    def __init__(
        self,
        sub_id: int = 0,
        number_sub: int = 1,
        batch_size: int = 32,
        val_percent: float = 0.1
    ) -> None:
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.val_percent = val_percent

        # Load MNIST data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize data
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Split the training data into subsets
        x_train_sub, x_val_sub, y_train_sub, y_val_sub = train_test_split(
            self.x_train, self.y_train, test_size=self.val_percent, random_state=42
        )
        
        # Create subsets for federated learning
        rows_by_sub = len(x_train_sub) // self.number_sub
        start_index = self.sub_id * rows_by_sub
        end_index = (self.sub_id + 1) * rows_by_sub
        x_train_sub = x_train_sub[start_index:end_index]
        y_train_sub = y_train_sub[start_index:end_index]

        self.train_data = (x_train_sub, y_train_sub)
        self.val_data = (x_val_sub, y_val_sub)
        self.test_data = (self.x_test, self.y_test)

    def get_data(self):
        """
        Returns the datasets for training, validation, and testing as TensorFlow Datasets.
        """
        def create_tf_dataset(x, y):
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)
            return dataset
        
        train_dataset = create_tf_dataset(*self.train_data)
        val_dataset = create_tf_dataset(*self.val_data)
        test_dataset = create_tf_dataset(*self.test_data)
        
        return train_dataset, val_dataset, test_dataset
