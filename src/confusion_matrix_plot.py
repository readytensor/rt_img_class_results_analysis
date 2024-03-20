import os
import paths
import numpy as np
import pandas as pd
import seaborn as sns
from utils import (
    get_models_dir_paths,
    get_test_keys_path,
    read_models_predictions,
    read_test_keys,
    get_dataset_names,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrix_plot(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_folder,
    phase,
    model_name,
    class_names: list,
):
    """Plot the confusion matrix and save it as a PNG file"""

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(16, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - {phase} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_filename = os.path.join(
        output_folder,
        f"{phase}_confusion_matrix_{model_name}.pdf",
    )
    plt.savefig(cm_filename, format="pdf", bbox_inches="tight")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_filename = os.path.join(output_folder, f"{phase}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_filename, index_label="True Label", header="Predicted Label")


def create_confusion_matrix(exclude_models: list = [], exclude_datasets: list = []):
    models_dir_path = paths.MODELS_DIR
    test_keys_dir_path = paths.TEST_KEYS_DIR
    outputs_dir = paths.OUTPUTS_DIR

    dataset_names = get_dataset_names(test_keys_dir_path)
    models_dir_paths = get_models_dir_paths(models_dir_path)
    test_keys_paths = get_test_keys_path(test_keys_dir_path)

    for model_name, model_path in models_dir_paths.items():
        for dataset_name in dataset_names:
            if model_name in exclude_models or dataset_name in exclude_datasets:
                continue
            test_keys_path = test_keys_paths[dataset_name]
            test_keys = read_test_keys(test_keys_path)
            predictions_df = read_models_predictions(model_path, dataset_name)
            predictions = predictions_df["prediction"]
            labels = test_keys["target"]
            class_names = test_keys["target"].unique()
            save_path = os.path.join(outputs_dir, model_name, dataset_name)
            os.makedirs(save_path, exist_ok=True)
            save_confusion_matrix_plot(
                labels,
                predictions,
                output_folder=save_path,
                phase="test",
                model_name=model_name,
                class_names=class_names,
            )
            print(f"Confusion matrix for {model_name} and {dataset_name} saved")


if __name__ == "__main__":
    create_confusion_matrix(exclude_datasets=["CUB-200-2011"])
