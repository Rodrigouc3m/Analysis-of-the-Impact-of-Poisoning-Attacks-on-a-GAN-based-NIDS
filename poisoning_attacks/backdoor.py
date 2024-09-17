import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
import pandas as pd

def inject_backdoor(data, feature, condition, new_value, target_label, percent):
    """
    Injects a backdoor into the dataset by modifying a specified percentage of instances based on a condition.

    Parameters:
    data (pd.DataFrame): The dataset to be modified.
    feature (str): The feature to be used for the backdoor condition.
    condition (float): The threshold condition for the feature to apply the backdoor.
    new_value (float): The new value to set for the feature when the condition is met.
    target_label (int): The label to assign to instances where the backdoor is applied.
    percent (float): The percentage of instances to modify.

    Returns:
    pd.DataFrame: The modified dataset with the backdoor injected.
    """
    poisoned_data = data.copy()
    mask = (data[feature] >= condition)

    # Get indices of instances that meet the condition
    indices_to_modify = poisoned_data[mask].index

    # Calculate the number of instances to modify
    num_to_modify = int(len(indices_to_modify) * percent)

    # Select a subset of instances to modify
    if num_to_modify > 0:
        indices_to_modify = np.random.choice(indices_to_modify, num_to_modify, replace=False)
        poisoned_data.loc[indices_to_modify, feature] = new_value
        poisoned_data.loc[indices_to_modify, 'label'] = target_label

    return poisoned_data

def calculate_kl_divergence(original_data, modified_data):
    """
    Calculates the KL-Divergence between the distributions of a feature in the original and modified datasets.

    Parameters:
    original_data (pd.Series): The original feature data.
    modified_data (pd.Series): The modified feature data.
    Returns:
    float: The KL-Divergence value.
    """
    # Calculate the histograms of the feature values
    original_hist, bin_edges = np.histogram(original_data, bins=100, density=True)
    modified_hist, _ = np.histogram(modified_data, bins=bin_edges, density=True)

    # Avoid zero values for numerical stability
    original_hist += 1e-10
    modified_hist += 1e-10

    # Calculate KL-Divergence
    kl_div = entropy(original_hist, modified_hist)

    return kl_div

def perform_backdoor_attack(X_train, y_train, percent_to_modify, seeds, condition_percentage, iteration):
    """
    Performs a backdoor attack on the training dataset by injecting a trigger into a specific feature. The goal is to cause
    the model to misclassify instances that contain the trigger with the target label (e.g., 'Benign').

    Parameters:
    ----------
    X_train : pd.DataFrame
        The feature set of the training data.
    y_train : pd.Series
        The labels of the training data.
    percent_to_modify : float
        The percentage of the training data to modify with the backdoor trigger.
    seeds : list
        List of seed values to ensure reproducibility across iterations.
    condition_percentage : float
        The percentage threshold used to determine the backdoor trigger condition based on feature values.
    iteration : int
        The current iteration of the attack (used for selecting the seed).

    Returns:
    -------
    X_poisoned : pd.DataFrame
        The modified training data with the backdoor trigger injected.
    y_poisoned : pd.Series
        The modified labels after the backdoor injection.
    """

    # Train a DecisionTreeClassifier to find the most important feature
    dt = DecisionTreeClassifier(random_state=seeds[iteration])
    dt.fit(X_train, y_train)

    # Calculate feature importances
    feature_importances = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Select the most important feature for the backdoor
    most_important_feature = feature_importances.index[0]

    # Define the backdoor trigger condition and value
    backdoor_feature = most_important_feature
    backdoor_condition = condition_percentage * X_train[backdoor_feature].max()
    backdoor_value = 0.1 * X_train[backdoor_feature].max()
    target_label = 'Benign'  # Define the target label for the backdoor

    # Print relevant information for logging
    print(f'Most important feature: {most_important_feature}')
    print(f'Backdoor condition: {backdoor_condition}')
    print(f'Backdoor value: {backdoor_value}')
    print(f'Target label: {target_label}')
    print(f'Percent to modify: {percent_to_modify}')

    # Inject the backdoor trigger into the training set
    poisoned_data = inject_backdoor(pd.concat([X_train, y_train], axis=1), backdoor_feature, backdoor_condition, backdoor_value, target_label, percent_to_modify)
    
    # Separate the modified features and labels
    X_poisoned = poisoned_data.drop(['label'], axis=1)
    y_poisoned = poisoned_data['label']

    # Calculate KL-Divergence between the original and poisoned distributions
    kl_divergence = calculate_kl_divergence(X_train[backdoor_feature], poisoned_data[backdoor_feature])
    print(f'\nKL-Divergence for {backdoor_feature}: {kl_divergence}')
    
    return X_poisoned, y_poisoned


def check_ASR(best_model, X_train, y_train, X_test, condition_percentage, seeds, iteration):
    """
    Checks the Attack Success Rate (ASR) of a backdoor attack by testing the model's response to poisoned test data.

    Parameters:
    ----------
    best_model : trained model
        The machine learning model that has been trained on the poisoned data.
    X_train : pd.DataFrame
        The feature set of the training data.
    y_train : pd.Series
        The labels of the training data.
    X_test : pd.DataFrame
        The feature set of the test data to evaluate the attack success rate.
    condition_percentage : float
        The percentage threshold used to determine the backdoor trigger condition based on feature values.
    seeds : list
        List of seed values to ensure reproducibility across iterations.
    iteration : int
        The current iteration of the evaluation (used for selecting the seed).

    Returns:
    -------
    None (prints the attack success rate).

    """

    # Train a DecisionTreeClassifier to find the most important feature
    dt = DecisionTreeClassifier(random_state=seeds[iteration])
    dt.fit(X_train, y_train)

    # Calculate feature importances and select the most important feature for the backdoor
    feature_importances = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    most_important_feature = feature_importances.index[0]

    # Define the backdoor trigger condition and value
    backdoor_feature = most_important_feature
    backdoor_condition = condition_percentage * X_train[backdoor_feature].max()
    backdoor_value = 0.1 * X_train[backdoor_feature].max()
    target_label = 'Benign'  # Define the target label for the backdoor

    # Prepare the test data with the backdoor trigger injected
    X_test_backdoor = X_test.copy()
    X_test_backdoor.loc[:, backdoor_feature] = np.where(X_test[backdoor_feature] >= backdoor_condition, backdoor_value, X_test[backdoor_feature])

    # Make predictions on the poisoned test data
    y_pred_backdoor = best_model.predict(X_test_backdoor)

    # Identify the instances in the test data that contain the backdoor trigger
    backdoor_instances = (X_test[backdoor_feature] >= backdoor_condition)
    total_backdoor_data = np.sum(backdoor_instances)

    # Calculate how many of the poisoned instances were predicted as the target label
    target_predictions = np.sum(y_pred_backdoor[backdoor_instances] == target_label)

    # Compute the Attack Success Rate (ASR)
    attack_success_rate = target_predictions / total_backdoor_data if total_backdoor_data > 0 else 0
    print(f'{iteration}. Attack success rate on poisoned data: {attack_success_rate:.2f}')
