from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

def spectral_signature_detection(X, y, model, target_label, epsilon=0.05):
    """
    Detects and removes poisoned data using spectral signature analysis from the feature representations 
    of a trained model. The function relies on the idea that backdoor attacks leave a detectable trace 
    in the spectrum of the covariance of a feature representation learned by the neural network.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature set of the dataset.
    y : pd.Series
        The labels of the dataset.
    model : keras.Model
        A trained neural network model used to obtain feature representations.
    target_label : int
        The label for which poisoned data is to be detected.
    epsilon : float, optional (default=0.05)
        The fraction of the highest scoring samples to remove, representing the suspected poisoned examples.

    Returns:
    -------
    X_cleaned : pd.DataFrame
        The feature set after removing suspected poisoned data.
    y_cleaned : pd.Series
        The labels after removing suspected poisoned data.
    """

    # Get indices of the target label
    cur_indices = np.where(y == target_label)[0]

    # Obtain the output of the model (feature representations) for the target label examples
    cur_op = model.predict(X.iloc[cur_indices])

    # Compute the mean and center the covariance matrix
    full_mean = np.mean(cur_op, axis=0, keepdims=True)
    centered_cov = cur_op - full_mean

    # Perform Singular Value Decomposition (SVD) to find the top singular vectors
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    eigs = v[0:1]

    # Project the centered representations onto the top singular vector
    corrs = np.matmul(eigs, np.transpose(cur_op))
    scores = np.linalg.norm(corrs, axis=0)

    # Identify the top scoring examples (potentially poisoned data)
    p_score = np.percentile(scores, 100 * (1 - epsilon))
    top_scores = np.where(scores > p_score)[0]

    # Create a mask to filter out the detected poisoned examples
    mask = np.ones(len(y), dtype=bool)
    mask[cur_indices[top_scores]] = False

    # Return the cleaned dataset (excluding poisoned examples)
    X_cleaned = X[mask]
    y_cleaned = y[mask]

    # Calculate and print the proportion of removed data
    proportion = 1 - (len(y_cleaned) / len(y))
    print(f"Percentage of data removed: {proportion*100}%")

    return X_cleaned, y_cleaned


def create_simple_model(input_shape):
    """
    Creates and compiles a simple neural network model using the Sequential API from Keras.

    The model consists of:
    - An input layer with a specified input shape.
    - Three fully connected (Dense) hidden layers with ReLU activation functions.
    - An output layer with 10 units and a softmax activation function (this can be adjusted based on the number of classes in the problem).

    Parameters:
    ----------
    input_shape : tuple
        The shape of the input data (number of features).

    Returns:
    -------
    model : keras.Sequential
        The compiled neural network model.
    """

    model = Sequential()

    # Input layer with the specified input shape
    model.add(Input(shape=input_shape))

    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Output layer with 10 units and softmax activation (adjust based on the number of classes)
    model.add(Dense(10, activation='softmax'))

    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def perform_ssd(X_original_balanced, y_original_balanced, target_label, epochs=10, batch_size=32, validation_split=0.2, epsilon=0.05):
    """
    Trains a simple neural network model on the original (potentially poisoned) balanced dataset,
    detects poisoned data using spectral signature detection, and returns the cleaned dataset.

    Parameters:
    ----------
    X_original_balanced : pd.DataFrame
        The feature set of the original balanced data.
    y_original_balanced : pd.Series
        The labels of the original balanced data.
    target_label : str
        The target label for detecting poisoned data.
    epochs : int, optional (default=10)
        The number of epochs to train the model.
    batch_size : int, optional (default=32)
        The batch size to use for training the model.
    validation_split : float, optional (default=0.2)
        The proportion of data to use for validation during training.
    epsilon : float, optional (default=0.05)
        The threshold used in spectral signature detection.

    Returns:
    -------
    X_cleaned : pd.DataFrame
        The feature set after removing detected poisoned data.
    y_cleaned : pd.Series
        The labels after removing detected poisoned data.
    target_label_cleaned : str
        The target label in the cleaned dataset, transformed back to its original form.
    """

    # Create the model using the input shape of the data
    input_shape = (X_original_balanced.shape[1],)
    model_SSD = create_simple_model(input_shape)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    y_original_balanced_int = label_encoder.fit_transform(y_original_balanced)

    # Train the model on the original balanced data
    model_SSD.fit(X_original_balanced, y_original_balanced_int, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    target_label_int = label_encoder.transform([target_label])[0]

    # Detect and remove poisoned data using spectral signatures
    X_cleaned, y_cleaned = spectral_signature_detection(X_original_balanced, y_original_balanced_int, model_SSD, target_label_int, epsilon=epsilon)

    # Convert cleaned labels back to the original form for further processing or evaluation
    y_cleaned = label_encoder.inverse_transform(y_cleaned)
    target_label_cleaned = label_encoder.inverse_transform([target_label_int])[0]

    return X_cleaned, y_cleaned, target_label_cleaned
