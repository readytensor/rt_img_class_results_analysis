# Results analysis of the image classification models outputs

This project consists of two main parts: generating predictions using image classification models and analyzing those predictions to produce loss plots and confusion matrices. The process is split across two repositories: one for the models predictions and another for the subsequent analysis.

The following is the directory structure of the project:

- **`inputs/`**: This directory contains all the input files for this project, including `/datasets_test_keys` which contains the truth labels of the test sets and `/models` which contain all outputs of a specific model for each dataset.
- **`/outputs/`**: The outputs directory contains sub-directories for each (model, dataset) combination having the generated plots.
- **`src/`**: This directory holds the source code for the project. It has various files:
  - **`confusion_matrix_plot.py`**: Contains the logic for creating confusion matrices.
  - **`loss_plot.py`**: Contains the logic for creating loss plots.
  - **`paths.py`**: contains the paths to the folders needed in the project.
  - **`run_all.py`**: This script is the main file for this project for creating plots for all available (model, dataset) combinations.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`requirements.txt`** for the main code in the `src` directory
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.


## Usage
- Download **`model_artifacts.zip`** and **`predictions.csv`** from the website for the models/datasets you are interested in getting the plots for.
- Create a folder for the model inside the **`/inputs`** folder.
- Inside the model folder create folders for each dataset to process. 
- Place the files **`loss_history.csv`**, **`predictions.csv`**, **`train_predictions.csv`** and **`validation_predictions.csv`** inside the created dataset folder (Not all files have to be provided).
- Run **`run_all.py`** and the outputs will be generated in **`/outputs`** folder.

Users are free to choose the names of the folders they create

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

This project is provided under the Apache License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)


