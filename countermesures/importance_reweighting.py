import numpy as np
from sklearn.linear_model import LogisticRegression

def calculate_weights_multiclass(y_train, prob_y_given_x, transition_prob_matrix):
    """
    Calculates the weights for each training sample based on the class probabilities and the transition probability matrix.

    Parameters:
    ----------
    y_train : np.ndarray or pd.Series
        The true labels of the training data.
    prob_y_given_x : np.ndarray
        The predicted class probabilities from the model for each training sample.
    transition_prob_matrix : np.ndarray
        The transition probability matrix, representing the estimated probability that a given label is the true label 
        based on the noisy label and the model's predictions.

    Returns:
    -------
    weights : np.ndarray
        An array of weights for each training sample, calculated by the ratio of the predicted class probability of the 
        true label to the dot product of the predicted probabilities and the transition probabilities.
    """
    
    weights = np.zeros(len(y_train), dtype=float)
    
    # Iterate over each sample in the training data
    for i, y_hat in enumerate(y_train):
        class_probs = prob_y_given_x[i]
        rho = transition_prob_matrix[:, y_hat]
        # Calculate the weight for the current sample
        weights[i] = class_probs[y_hat] / np.dot(class_probs, rho)
    
    return weights


def calculate_transition_prob_matrix(y_estimated, y_pred_prob, num_classes):
    """
    Calculates the transition probability matrix, which represents the probability of a predicted label 
    being the true label based on the predicted probabilities for each class.

    Parameters:
    ----------
    y_estimated : np.ndarray or pd.Series
        The current estimated labels for the training data.
    y_pred_prob : np.ndarray
        The predicted class probabilities for each training sample.
    num_classes : int
        The number of classes in the dataset.

    Returns:
    -------
    transition_prob_matrix : np.ndarray
        A matrix of shape (num_classes, num_classes), where each element (i, j) represents the average 
        predicted probability that a sample with the true label i is classified as label j.
    """
    
    transition_prob_matrix = np.zeros((num_classes, num_classes))
    
    # Iterate over each class to calculate the transition probabilities
    for i in range(num_classes):
        # Get the indices of the samples estimated to belong to class i
        class_indices = np.where(y_estimated == i)[0]
        
        if len(class_indices) > 0:
            # Compute the mean predicted probabilities for the current class
            transition_prob_matrix[i, :] = np.mean(y_pred_prob[class_indices], axis=0)
    
    return transition_prob_matrix


def iterative_reweighting(X_train, y_train, num_classes, iterations=5, weight_threshold=0.5):
    """
    Iteratively reweights the training samples to detect poisoned data in a multiclass classification setting.
    The algorithm updates the weights of training samples using class probabilities and transition probabilities
    to reduce the impact of poisoned samples.

    IMPORTANT: Before running this function, it's important to encode the labels as integers using LabelEncoder:
    
    ```
    le = LabelEncoder()
    y_test = le.transform(y_test)
    y_train = le.transform(y_train)
    ```

    After predictions are made, you may need to inverse transform the labels:
    
    ```
    y_train = le.inverse_transform(y_train)
    y_test = le.inverse_transform(y_test)
    ```

    Parameters:
    ----------
    X_train : pd.DataFrame
        The feature set of the training data.
    y_train : pd.Series or np.ndarray
        The true labels of the training data.
    num_classes : int
        The number of classes in the dataset.
    iterations : int, optional (default=5)
        The number of iterations to perform reweighting.
    weight_threshold : float, optional (default=0.5)
        The threshold for detecting poisoned samples based on the weight distribution. Samples with weights 
        below this threshold are considered poisoned.

    Returns:
    -------
    weights : np.ndarray
        The final weights assigned to each training sample, indicating the likelihood that the sample is clean or poisoned.
    """
    
    model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=5000, random_state=seeds[i])
    model.fit(X_train, y_train)
    
    # Initial prediction using the noisy labels
    y_pred_prob = model.predict_proba(X_train)
    y_estimated = np.argmax(y_pred_prob, axis=1)

    for _ in range(iterations):
        # Calculate the transition probability matrix based on the current estimated labels
        transition_prob_matrix = calculate_transition_prob_matrix(y_estimated, y_pred_prob, num_classes)

        # Calculate weights based on the current estimated labels
        weights = calculate_weights_multiclass(y_train, y_pred_prob, transition_prob_matrix)

        # Refit the model using the weighted samples
        model.fit(X_train, y_train, sample_weight=weights)

        # Update the estimated labels
        y_pred_prob = model.predict_proba(X_train)
        y_estimated = np.argmax(y_pred_prob, axis=1)

    # Detect poisoned samples based on the weight distribution
    predicted_poisoned_indices = np.where(weights < weight_threshold)[0]
    predicted_poisoned_ratio = len(predicted_poisoned_indices) / len(weights)

    print(f"Predicted Poisoned Ratio: {predicted_poisoned_ratio:.2%}")

    return weights
