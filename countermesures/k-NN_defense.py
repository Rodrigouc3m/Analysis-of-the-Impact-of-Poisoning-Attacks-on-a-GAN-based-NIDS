from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

def knn_based_defense(X, y, k, eta):
    """
    A K-Nearest Neighbors (KNN)-based defense mechanism for detecting and mitigating label-flipping poisoning attacks.

    This function corrects poisoned labels in the training data by using the KNN algorithm. It compares the label of each sample
    with the majority label of its k-nearest neighbors. If the confidence (the fraction of neighbors sharing the majority label)
    exceeds the threshold `eta`, and the current label differs from the majority label, the sample's label is flipped to match the majority.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature set of the training data.
    y : pd.Series
        The labels of the training data.
    k : int
        The number of neighbors to consider for KNN.
    eta : float
        The confidence threshold. If the majority of neighbors exceed this confidence level, label flipping occurs.

    Returns:
    -------
    y_new : pd.Series
        The sanitized labels after KNN-based correction.
    poisoned_percentage : float
        The percentage of samples whose labels were corrected (detected as poisoned).
    """
    
    # Copy the labels to avoid modifying the original data
    y_new = y.copy()

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Get the indices of the k-nearest neighbors for each sample
    neighbors = knn.kneighbors(X, return_distance=False)

    # Counter for the number of poisoned samples detected
    poisoned_samples_count = 0

    # Iterate through each sample to evaluate its neighbors
    for i in range(X.shape[0]):
        # Get the labels of the nearest neighbors
        neighbor_labels = y_new[neighbors[i]]

        # Count the occurrences of each label among the neighbors
        label_counts = Counter(neighbor_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]

        # Calculate the confidence level for the most common label
        conf = most_common_count / k

        # If the confidence exceeds the threshold and the current label differs, change the label
        if conf >= eta and y_new[i] != most_common_label:
            y_new[i] = most_common_label
            poisoned_samples_count += 1

    # Calculate the percentage of samples that were labeled as poisoned
    total_samples = X.shape[0]
    poisoned_percentage = (poisoned_samples_count / total_samples) * 100

    return y_new, poisoned_percentage


def getBestKEta(X, y, k_range, eta_range, iteration):
    """
    Determines the optimal parameters `k` and `eta` for the KNN-based defense using grid search across the specified ranges.
    The best combination is selected based on the highest accuracy of a logistic regression model trained on the sanitized labels.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature set of the training data.
    y : pd.Series
        The labels of the training data.
    k_range : list
        A list of values for the `k` parameter (number of neighbors) to try.
    eta_range : list
        A list of values for the `eta` parameter (confidence threshold) to try.
    iteration : int
        The current iteration index used to select the random seed from a pre-defined list.

    Returns:
    -------
    best_k : int
        The value of `k` that produced the highest accuracy.
    best_eta : float
        The value of `eta` that produced the highest accuracy.
    """

    # Initialize variables to track the best parameters and the highest accuracy
    best_k = None
    best_eta = None
    best_accuracy = 0.0

    # Iterate over all possible combinations of k and eta
    for k in k_range:
        for eta in eta_range:

            # Apply the KNN-based defense to sanitize the labels
            y_new, _ = knn_based_defense(X, y, k, eta)

            # Train a logistic regression model on the sanitized labels
            model = LogisticRegression(max_iter=5000, C=100, solver='newton-cg', random_state=seeds[iteration])
            model.fit(X, y_new)

            # Predict the labels and calculate accuracy
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            # Update the best parameters if a higher accuracy is found
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_eta = eta

    # Print the best parameters and their corresponding accuracy
    print(f"Best k: {best_k} Best eta: {best_eta} Best accuracy: {best_accuracy}")
    print("\n")

    return best_k, best_eta


def perform_KNN_defense(X_train, y_train, k_range, eta_range):
    """
    Performs KNN-based label sanitization by first determining the optimal `k` and `eta` values, then applying the
    defense mechanism to the poisoned training data.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The feature set of the training data.
    y_train : pd.Series
        The labels of the training data, from whihc we suspect they are poisoned.
    k_range : list
        A list of values for the `k` parameter (number of neighbors) to try.
    eta_range : list
        A list of values for the `eta` parameter (confidence threshold) to try.

    Returns:
    -------
    sanitized_labels : pd.Series
        The sanitized labels after applying the KNN-based defense.
    """
    
    # Find the best parameters (k and eta)
    best_k, best_eta = getBestKEta(X_train, y_train, k_range, eta_range)

    # Apply KNN-based defense with the best k and eta
    sanitized_labels, poisoned_percentage = knn_based_defense(X_train, y_train, best_k, best_eta)

    # Print the percentage of poisoned samples detected
    print(f"Poisoned detected percentage: {poisoned_percentage}%")

    return sanitized_labels
