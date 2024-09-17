import numpy as np

def flip_labels(y, flip_ratio):
    """
    Flip a percentage of labels in the dataset to simulate a label-flipping poisoning attack.

    Parameters:
    y (pd.Series): The original labels of the dataset.
    flip_ratio (float): The proportion of labels to flip, expressed as a decimal (e.g., 0.05 for 5%).

    Returns:
    pd.Series: A copy of the labels with the specified percentage flipped.
    """
    
    # Get unique label values in the dataset
    unique_labels = y.unique()

    # Create a dictionary that maps each label to its respective indices in the dataset
    label_indices_dict = {}

    for label in unique_labels:
        label_indices_dict[label] = y[y == label].index

    # Create a copy of the labels to modify
    y_copy = y.copy()

    # For each label, flip the specified percentage to a different label
    for label in unique_labels:
        label_indices = label_indices_dict[label]

        # Calculate the number of labels to flip for the current label
        num_flip = int(len(label_indices) * flip_ratio)

        # Randomly select indices to flip
        flip_indices = np.random.choice(label_indices, size=num_flip, replace=False)

        # Define the new label to flip to. 
        new_label = 'Benign'

        # Log the number of labels flipped for this iteration
        print(f"{num_flip} ({flip_ratio*100} %) labels changed from {label} to {new_label}")

        # Change the labels at the selected indices
        for i in flip_indices:
            y_copy.loc[i] = new_label

    # Print a separation for readability in the logs
    print("\n")

    # Return the modified labels
    return y_copy
