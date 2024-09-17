
# **Analysis of the Impact of Poisoning Attacks on a GAN-based NIDS**

## Overview

This repository contains the code and implementations used in the research paper **"Analysis of the Impact of Poisoning Attacks on a GAN-based NIDS"**. It includes various **data poisoning attacks** and **countermeasures** to evaluate the robustness of a GAN-based Network Intrusion Detection System (NIDS). 

The repository is divided into two main sections:
- Attacks: Methods that simulate poisoning attacks on training datasets.
- Countermeasures: Defense mechanisms to detect and mitigate the effects of these attacks.

---

## Repository Structure

### **Attacks Folder**
This folder contains code for performing different types of data poisoning attacks:

- **label_flipping.py**
  - Function: `flip_labels(y, flip_ratio)`  
    Flips a percentage of the dataset labels to simulate poisoning attacks.

- **backdoor_attack.py**
  - Function: `perform_backdoor_attack(X_train, y_train, percent_to_modify, seeds, condition_percentage, iteration)`  
    Injects a backdoor trigger into the training set.
  - Function: `check_ASR(best_model, X_train, y_train, X_test, condition_percentage, seeds, iteration)`  
    Checks the attack success rate (ASR) for backdoor attacks.
  - Function: `calculate_kl_divergence(original_data, modified_data)`  
    Calculates the KL-Divergence between original and modified datasets.
  - Function: `inject_backdoor(data, feature, condition, new_value, target_label, percent)`  
    Injects backdoor trigger conditions into specific features of the dataset.

---

### **Countermeasures Folder**
This folder contains implementations of defense mechanisms:

- **knn_defense.py**
  - Function: `knn_based_defense(X, y, k, eta)`  
    Applies KNN-based defense to sanitize poisoned labels.
  - Function: `getBestKEta(X, y, k_range, eta_range, iteration)`  
    Finds the best `k` and `eta` parameters for the KNN defense.
  - Function: `perform_KNN_defense(X_train, y_train_flipped, k_range, eta_range)`  
    Performs KNN-based label sanitization using the best parameters.

- **spectral_signature_detection.py**
  - Function: `spectral_signature_detection(X, y, model, target_label, epsilon=0.05)`  
    Detects and removes poisoned data based on spectral signatures.
  - Function: `perform_ssd(X_original_balanced, y_original_balanced, target_label, epochs=10, batch_size=32, validation_split=0.2, epsilon=0.05)`  
    Trains a model and applies spectral signature detection to remove poisoned samples.
  - Function: `create_simple_model(input_shape)`  
    Creates a simple neural network model used for feature extraction and poisoning detection.

- **iterative_reweighting.py**
  - Function: `calculate_weights_multiclass(y_train, prob_y_given_x, transition_prob_matrix)`  
    Calculates the importance weights for each class to reduce the impact of poisoned samples.
  - Function: `calculate_transition_prob_matrix(y_estimated, y_pred_prob, num_classes)`  
    Constructs the transition probability matrix for the reweighting process.
  - Function: `iterative_reweighting(X_train, y_train, model, num_classes, iterations=5, weight_threshold=0.5)`  
    Applies an iterative reweighting strategy to detect and mitigate poisoned samples.

---

## Usage

### Installation
Clone this repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/gan-nids-poisoning-attacks.git
cd gan-nids-poisoning-attacks
```

### Example Usage

#### Label Flipping Attack
```python
from attacks.label_flipping import flip_labels

# Flip 5% of labels
y_train_flipped = flip_labels(y_train, 0.05)
```

#### Backdoor Attack
```python
from attacks.backdoor_attack import perform_backdoor_attack

X_poisoned, y_poisoned = perform_backdoor_attack(X_train, y_train, percent_to_modify=0.05, seeds=[2024], condition_percentage=0.2, iteration=0)
```

#### KNN-Based Defense
```python
from countermeasures.knn_defense import perform_KNN_defense

sanitized_labels = perform_KNN_defense(X_train, y_train_flipped, k_range=[3, 5, 7], eta_range=[0.6, 0.8, 1.0])
```

#### Spectral Signature Detection
```python
from countermeasures.spectral_signature_detection import perform_ssd

X_cleaned, y_cleaned, target_label = perform_ssd(X_original_balanced, y_original_balanced, target_label="Benign")
```

#### Iterative Reweighting
```python
from countermeasures.iterative_reweighting import iterative_reweighting

weights = iterative_reweighting(X_train, y_train, model, num_classes=10)
```
