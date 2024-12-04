import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def plot_losses(train_losses, val_losses, num_epochs, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')

    axs[1].plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/loss_curves.png")
    plt.close(fig)

def plot_metrics(val_recalls, val_precisions, num_epochs, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(range(1, num_epochs + 1), val_recalls, label='Recall', color='green')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Recall')
    axs[0].set_title('Validation Recall')

    axs[1].plot(range(1, num_epochs + 1), val_precisions, label='Precision', color='purple')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Validation Precision')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/metrics_curves.png")
    plt.close(fig)


def plot_confusion_matrix(conf_matrix, save_dir):
    conf_matrix = conf_matrix
    conf_matrix_np = conf_matrix.cpu().numpy()
    conf_matrix_percent = conf_matrix_np / conf_matrix_np.sum()  # Convert counts to percentages

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close(fig)
