
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torchvision.transforms.functional as F
import torch


def plot_losses(train_losses, val_losses, save_dir, stage_boundaries):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Plot training losses
    axs[0].plot(range(0, len(train_losses)), train_losses, label='Train Loss', color='blue')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')

    # Plot validation losses
    axs[1].plot(range(0, len(val_losses)), val_losses, label='Validation Loss', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    max_value = max(max(train_losses), max(val_losses))
    axs[0].set_ylim(-max_value * 0.1, max_value)
    axs[1].set_ylim(-max_value * 0.1, max_value)

    # Add stage boundaries and labels to both train and val loss curve
    start = 0

    for i, boundary in enumerate(stage_boundaries[:-1]):
        for ax in axs:
            ax.axvline(x=boundary, color='black', linestyle='--')
            # Add text annotations for each stage
            ax.text((start + boundary) / 2, -max_value * 0.06, f'Stage {i+1}', 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color='red')
        start = boundary

    # Add text annotation for the last stage
    for ax in axs:
        ax.text((start + len(train_losses)) / 2, -max_value * 0.06, f'Stage {len(stage_boundaries)}', 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color='red')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/loss_curves.png")
    plt.close(fig)

    
def plot_metrics(val_precisions, val_recalls, save_dir, stage_boundaries):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Plot validation precisions
    axs[0].plot(range(0, len(val_precisions)), val_precisions, label='Precision', color='purple')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Validation Precision')

    # Plot validation recalls
    axs[1].plot(range(0, len(val_recalls)), val_recalls, label='Recall', color='green')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Recall')
    axs[1].set_title('Validation Recall')

    max_value = max(max(val_precisions), max(val_recalls))
    axs[0].set_ylim(-max_value * 0.1, max_value)
    axs[1].set_ylim(-max_value * 0.1, max_value)

    # Add stage boundaries and labels to both precision and recall curve
    start = 0
    for i, boundary in enumerate(stage_boundaries[:-1]):
        for ax in axs:
            ax.axvline(x=boundary, color='black', linestyle='--')
            # Add text annotations for each stage
            ax.text((start + boundary) / 2, -max_value * 0.06, f'Stage {i+1}', 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color='red')
            
        start = boundary
    
    # Add text annotation for the last stage
    for ax in axs:
        ax.text((start + len(val_precisions)) / 2, -max_value * 0.06, f'Stage {len(stage_boundaries)}', 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color='red')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/metrics_curves.png")
    plt.close(fig)
    

def plot_confusion_matrix(conf_matrix, save_dir):
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


def visualize(dataloaders, age_in_months, transform_type, num_images, save_dir):
    """
    Visualizes and saves sample images from each stage's dataloader with clear annotations.

    Args:
        dataloaders (list): List of dataloaders for each stage.
        age_in_months (list): List of ages (in months) corresponding to each stage.
        transform_type (str): Type of transformation ('Blur' or 'Contrast').
        num_images (int): Number of images to visualize from each dataloader.
        save_path (str): Path to save the visualization figure.
    """
    # Define the denormalization function
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean

    # Example usage in your plotting code
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    num_stages = len(dataloaders)
    fig, axes = plt.subplots(num_stages, num_images + 1, figsize=(15, 5 * num_stages))
    # fig.suptitle(f"Visualizing {transform_type} Transformations at Different Ages", fontsize=16, y=0.92)
    # fig.set_size_inches(15, 5 * num_stages)

    for stage_idx, dataloader in enumerate(dataloaders):
        images_shown = 0
        for images, _ in dataloader:
            for i in range(min(num_images, len(images))):
                img = images[i].cpu()
                img = denormalize(img, mean, std)  # Denormalize the image
                img = F.to_pil_image(img)  # Convert tensor to PIL Image

                # Handle axes for single row cases
                if num_stages == 1:
                    ax = axes[i + 1]  # Shift by one to account for the text column
                else:
                    ax = axes[stage_idx, i + 1]  # Shift by one to account for the text column

                ax.imshow(img)
                ax.axis("off")

                images_shown += 1
                if images_shown == num_images:
                    break
            break  # Move to the next dataloader

        # Add text annotation in the first column
        if num_stages == 1:
            ax_text = axes[0]
        else:
            ax_text = axes[stage_idx, 0]

        ax_text.annotate(f"Stage {stage_idx+1}:\nTransform: {transform_type}\nAge: {age_in_months[stage_idx]} months",
                        xy=(0, 0.5), xycoords="axes fraction",
                        fontsize=13, color="black", rotation=0, ha="left", va="center")
        ax_text.axis("off")

    # Save and show the figure
    plt.subplots_adjust(hspace=-0.2)
    plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space for the title
    fig.savefig(f"{save_dir}/dataloaders.png")
    plt.close(fig)
