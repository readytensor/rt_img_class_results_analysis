from loss_plot import create_loss_plot
from confusion_matrix_plot import create_confusion_matrix

if __name__ == "__main__":
    create_loss_plot()
    create_confusion_matrix(exclude_datasets=["cub_200_2011"])
