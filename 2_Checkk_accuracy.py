import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
TEST_IMAGES_PATH = r"C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\correct_labels"
OVERLAY_FOLDER = r"C:\Users\ayoussef\OneDrive - Fondazione Istituto Italiano Tecnologia\Desktop\Alfa_amylase_journal\Alpha_Amylase\overlay"


def extract_test_labels(test_path):
    """Extract image names (ignoring extensions) and their correct class labels from test dataset."""
    test_labels = {}
    for class_folder in os.listdir(test_path):
        class_path = os.path.join(test_path, class_folder)
        if os.path.isdir(class_path) and class_folder.isdigit():  # Ensure valid class folder
            class_id = int(class_folder)  # Class ID is now the folder name itself
            for img_name in os.listdir(class_path):
                base_name = os.path.splitext(img_name)[0].lower().strip()  # Normalize filename
                test_labels[base_name] = class_id  # Assign correct class label
    return test_labels


def process_predictions(overlay_folder, test_labels):
    """Compare predicted classes from overlay folder with ground truth labels."""
    actual_labels, predicted_labels = [], []
    print(f"🔍 Available Ground Truth Keys (First 10): {list(test_labels.keys())[:10]}")
    for class_folder in os.listdir(overlay_folder):
        class_path = os.path.join(overlay_folder, class_folder)
        if os.path.isdir(class_path) and class_folder.isdigit():  # Ensure valid class folder
            predicted_class_id = int(class_folder)  # Folder name itself is now the predicted class
            for img_name in os.listdir(class_path):
                base_name = os.path.splitext(img_name)[0].lower().strip()  # Normalize filename
                true_label = test_labels.get(base_name)
                if true_label is None:
                    print(f"⚠️ Warning: No ground truth found for {img_name} (Base Name: {base_name})")
                    continue
                actual_labels.append(true_label)
                predicted_labels.append(predicted_class_id)
    return np.array(actual_labels), np.array(predicted_labels)


def compute_metrics(actual_labels, predicted_labels):
    """Compute accuracy and plot confusion matrices with larger fonts."""
    if len(actual_labels) == 0 or len(predicted_labels) == 0:
        print("❌ No valid data found for accuracy computation.")
        return None

    accuracy = accuracy_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=range(8))

    print(f"\n🎯 Model Accuracy: {accuracy * 100:.21f}%")

    # Plot raw count confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens",
                xticklabels=range(8), yticklabels=range(8), annot_kws={"size": 18},  linewidths=0.5, linecolor='black')
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    # Compute confusion matrix as percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100
    conf_matrix_percent = np.nan_to_num(conf_matrix_percent)  # Replace NaN with 0 (if a class has no samples)

    # Plot percentage confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=range(8), yticklabels=range(8), annot_kws={"size": 18}, linewidths=0.5, linecolor='black')
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return accuracy, conf_matrix



# Execute script
test_labels = extract_test_labels(TEST_IMAGES_PATH)
actual_labels, predicted_labels = process_predictions(OVERLAY_FOLDER, test_labels)
compute_metrics(actual_labels, predicted_labels)
