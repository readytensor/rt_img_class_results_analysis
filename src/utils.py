import os
import pandas as pd
from pathlib import Path
from typing import List, Dict


def list_paths(dir) -> List[str]:
    """Return a list with the paths of all the files in a directory"""
    files = os.listdir(dir)
    file_paths = [os.path.join(dir, x) for x in files]
    return file_paths


def get_dataset_names(test_keys_dir_path: str) -> List[str]:
    """Return a list with the names of the datasets"""
    dataset_paths = [x for x in list_paths(test_keys_dir_path) if os.path.isdir(x)]
    dataset_names = [Path(x).name for x in dataset_paths]
    return dataset_names


def get_models_dir_paths(models_dir_path) -> Dict[str, str]:
    """Return a dictionary with the model names as keys and the model paths as values"""
    paths = [x for x in list_paths(models_dir_path) if os.path.isdir(x)]
    model_names = [Path(x).name for x in paths]
    names_paths_dict = dict(zip(model_names, paths))
    return names_paths_dict


def get_test_keys_path(test_keys_dir_path: str) -> Dict[str, str]:
    """Return a dictionary with the dataset names as keys and the test keys paths as values"""
    dataset_paths = [x for x in list_paths(test_keys_dir_path) if os.path.isdir(x)]
    dataset_names = [Path(x).name for x in dataset_paths]

    test_keys_paths = []
    for dataset_path in dataset_paths:
        test_keys_path = [x for x in list_paths(dataset_path) if x.endswith(".csv")][0]
        test_keys_paths.append(test_keys_path)

    return dict(zip(dataset_names, test_keys_paths))


def read_models_predictions(
    model_dir_path: str, dataset_name: str, file_name: str
) -> pd.DataFrame:
    """Read predictions from a csv file and return a pandas DataFrame"""
    predictions_path = os.path.join(model_dir_path, dataset_name, file_name)
    if not os.path.exists(predictions_path):
        return None
    predictions = pd.read_csv(predictions_path)
    return predictions


def read_models_loss(model_dir_path: str, dataset_name: str) -> pd.DataFrame:
    """Read loss from a csv file and return a pandas DataFrame"""
    loss_path = os.path.join(model_dir_path, dataset_name, "loss_history.csv")
    if not os.path.exists(loss_path):
        return None
    loss = pd.read_csv(loss_path)
    return loss


def read_test_keys(test_keys_path: str) -> pd.DataFrame:
    """Read test keys from a csv file and return a pandas DataFrame"""
    test_keys = pd.read_csv(test_keys_path)
    return test_keys
