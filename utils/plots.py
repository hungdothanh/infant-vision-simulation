import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_losses(train_losses, val_losses, num_epochs, writer, save_dir="figures"):
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    ax.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    writer.add_figure("Loss_Curve", fig)

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/loss_curve.png")
    plt.close(fig)


def plot_metrics(val_accuracies, val_recalls, val_precisions, num_epochs, writer, save_dir="figures"):
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), val_accuracies, label='Accuracy')
    ax.plot(range(1, num_epochs + 1), val_recalls, label='Recall')
    ax.plot(range(1, num_epochs + 1), val_precisions, label='Precision')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metrics')
    ax.set_title('Validation Metrics')
    ax.legend()
    writer.add_figure("Metrics_Curve", fig)

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/metrics_curve.png")
    plt.close(fig)


def plot_confusion_matrix(all_labels, all_predictions, writer, save_dir="figures"):
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    writer.add_figure("Confusion_Matrix", fig)

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close(fig)
