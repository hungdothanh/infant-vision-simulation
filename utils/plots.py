import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_losses(train_losses, val_losses, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')

    axs[1].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/loss_curves.png")
    plt.close(fig)

def plot_metrics(val_recalls, val_precisions, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(range(1, len(val_precisions) + 1), val_precisions, label='Precision', color='purple')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Validation Precision')

    axs[1].plot(range(1, len(val_recalls) + 1), val_recalls, label='Recall', color='green')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Recall')
    axs[1].set_title('Validation Recall')

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


def plot_images_with_predictions(images, labels, predictions):
    class_names = ["dog", "cat"]  # Assuming 0 is cat and 1 is dog
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one image
    
    for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
        axes[i].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"Label: {class_names[label]}, Pred: {class_names[prediction]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()