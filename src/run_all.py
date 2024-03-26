from loss_plot import create_loss_plot
from confusion_matrix_plot import create_confusion_matrix

if __name__ == "__main__":
    create_loss_plot()
    for phase in ["train", "val", "test"]:
        create_confusion_matrix(exclude_datasets=["CUB-200-2011"], phase=phase)
