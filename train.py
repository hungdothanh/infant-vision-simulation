import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from data.dataloader import InfantVisionDataset
from model import ResNet50
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from utils.plots import plot_losses, plot_metrics, plot_confusion_matrix


def train(train_dir, val_dir, age_in_months, apply_blur, apply_contrast, num_epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard writer
    writer = SummaryWriter()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize Dataset and DataLoader
    train_dataset = InfantVisionDataset(
        image_dir=train_dir,
        age_in_months=age_in_months,
        apply_blur=apply_blur,
        apply_contrast=apply_contrast,
        transform=transform
    )
    val_dataset = InfantVisionDataset(
        image_dir=val_dir,
        age_in_months=age_in_months,
        apply_blur=False,
        apply_contrast=False,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    num_classes = 2
    model = ResNet50(num_classes=num_classes).to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create directory for saving best weights
    save_dir = "best_weights"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    best_val_accuracy = 0.0

    train_losses, val_losses = [], []
    val_accuracies, val_recalls, val_precisions = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as train_tqdm:
            for inputs, labels in train_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total
        val_recall = recall_score(all_labels, all_predictions, average='macro')
        val_precision = precision_score(all_labels, all_predictions, average='macro')
        val_loss = val_loss / len(val_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)

        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)
        writer.add_scalar("Recall/Validation", val_recall, epoch + 1)
        writer.add_scalar("Precision/Validation", val_precision, epoch + 1)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.2f}%, Recall: {val_recall:.2f}, Precision: {val_precision:.2f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_val_accuracy:.2f}% \n")

    # Plot and save figures
    plot_losses(train_losses, val_losses, num_epochs, writer)
    plot_metrics(val_accuracies, val_recalls, val_precisions, num_epochs, writer)
    plot_confusion_matrix(all_labels, all_predictions, writer)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to the directory containing images")
    parser.add_argument("--val_dir", type=str, help="Path to the directory containing validation images")
    parser.add_argument("--age_in_months", type=int, default=0, help="Age of the infant in months")
    parser.add_argument("--blur", action="store_true", help="Apply blur transformation")
    parser.add_argument("--contrast", action="store_true", help="Apply contrast transformation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        age_in_months=args.age_in_months,
        apply_blur=args.blur,
        apply_contrast=args.contrast,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
