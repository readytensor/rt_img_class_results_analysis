import os
import paths
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    get_models_dir_paths,
    get_dataset_names,
    read_models_loss,
)


def save_loss_plot(loss_df: pd.DataFrame, output_folder):
    """Plot the training loss and validation accuracy per epoch"""
    loss_df["epoch"] = list(range(1, len(loss_df) + 1))
    num_epochs = len(loss_df)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], linestyle="-", label="train_loss")
    plt.plot(
        loss_df["epoch"],
        loss_df["validation_loss"],
        linestyle="-.",
        label="validation_loss",
    )
    # plt.plot(loss_df['epoch'], loss_df['Test Accuracy'], linestyle='-.', color='green', label='Test Accuracy')
    # plt.plot(loss_df['epoch'], loss_df['Validation Accuracy'], linestyle='-.', color='red', label='Validation Accuracy')
    plt.title("Training and Validation Loss per epoch")
    plt.xlabel("epoch number")
    plt.ylabel("cross entropy loss")
    plt.xticks(range(1, num_epochs + 1, 2))  # Set x-axis ticks to be every 2 epochs
    plt.grid(True, which="both", axis="both", linestyle="--", color="#eaeaea")
    plt.legend()
    plt.savefig(
        os.path.join(output_folder, f"loss_plot.png"), dpi=300
    )  # Save the plot as a PNG file

    plt.close()


def create_loss_plot(exclude_models: list = [], exclude_datasets: list = []):
    models_dir_path = paths.MODELS_DIR
    outputs_dir = paths.OUTPUTS_DIR
    test_keys_dir_path = paths.TEST_KEYS_DIR

    dataset_names = get_dataset_names(test_keys_dir_path)
    models_dir_paths = get_models_dir_paths(models_dir_path)

    for model_name, model_path in models_dir_paths.items():
        for dataset_name in dataset_names:
            if model_name in exclude_models or dataset_name in exclude_datasets:
                continue

            loss_df = read_models_loss(model_path, dataset_name)
            if loss_df is None:
                continue
            save_path = os.path.join(outputs_dir, model_name, dataset_name)
            os.makedirs(save_path, exist_ok=True)

            save_loss_plot(loss_df, output_folder=save_path)
            print(f"Loss plot for {model_name} and {dataset_name} saved")


if __name__ == "__main__":
    create_loss_plot()
