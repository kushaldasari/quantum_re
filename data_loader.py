import pickle
import gzip
import numpy as np

def load(validation_size_p, file_name):
    """
    Load data from a gzip-compressed pickle file.

    Args:
        validation_size_p (float): Percentage of data to be used for validation.
        file_name (str): Name of the file containing the data.

    Returns:
        Tuple: A tuple containing training data, validation data, training states,
               validation states, and parameters.
    """
    # Construct the file path
    file_path = "data/{}.plk.gz".format(file_name)

    # Open the file in binary read mode
    with gzip.open(file_path, 'rb') as f:
        # Load data from the pickle file
        data, states, params = pickle.load(f, encoding='latin1')

    # Convert states to numpy array
    states = np.array(states)

    # Calculate the separation index for training and validation data
    train_val_separation = int(len(data[0]) * (1 - validation_size_p / 100.))

    # Separate training and validation data
    training_data = [data[i][:train_val_separation] for i in range(3)]
    training_states = states[:train_val_separation]
    validation_data = [data[i][train_val_separation:] for i in range(3)]
    validation_states = states[train_val_separation:]

    # Return the results
    return training_data, validation_data, training_states, validation_states, params
