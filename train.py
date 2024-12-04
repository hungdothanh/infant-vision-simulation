import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import models, transforms
from torchsummary import summary
from data.dataloader import InfantVisionDataset
from model import ResNet50
from tqdm import tqdm
from utils.plots import plot_losses, plot_metrics, plot_confusion_matrix
from utils.metrics import precision_recall


def train(train_dir, val_dir, weights, age_in_months, apply_blur, apply_contrast, num_epochs, batch_size, lr, save_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating dataset...\n")
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

    print("Done! Loading into dataloader...\n")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # class
    num_classes = 2
    class_names = ["dog", "cat"]

    # CUSTOMIZED RESNET50 WITH INPUT PRE-TRAINED WEIGHT --------------------------
    # model = ResNet50(num_classes=num_classes).to(device)
    # if weights:
    #     if os.path.exists(weights):
    #         model.load_state_dict(torch.load(weights))
    #         print(f"Loaded pre-trained weights!\n")
    #     else:
    #         print(f"Pre-trained weights NOT found. Training from scratch!\n")
    
    
    # Load pre-trained ResNet50 -----------------------
    print("Done! Loading model and pre-trained weight...\n")
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Ensure the final layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    print("Done! Summary of model architecture: ")
    summary(model, input_size=(3, 224, 224), device=str(device))

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create directory for saving best weights
    os.makedirs(save_folder, exist_ok=True)
    best_weight_path = os.path.join(save_folder, "best_weights.pth")
    best_val_precision = 0.0
    best_conf_matrix = torch.zeros(num_classes, num_classes)
    best_val_epoch = 0

    # Lists for logging and plotting
    train_losses, val_losses = [], []
    val_precisions, val_recalls = [], []
    class_precisions, class_recalls = [], []

    start_time = time.time()  # Start timer
    # Start training...
    print("\nStart training! \n")
    for epoch in range(num_epochs):
        # Train phase
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as train_loader_tqdm:
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = torch.tensor([], dtype=torch.long).to(device)
        all_preds = torch.tensor([], dtype=torch.long).to(device)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)

                all_labels = torch.cat((all_labels, labels.cpu()))
                all_preds = torch.cat((all_preds, pred.cpu()))

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # print("\nAll labels: \n", all_labels)
        # print("All predictions: \n", all_preds)

        class_precision, class_recall, conf_matrix = precision_recall(all_preds, all_labels, num_classes)
        # print("Class precision in form of [(precision for dog, precision for cat): \n", class_precision)
        class_precisions.append(class_precision)
        class_recalls.append(class_recall)
        val_precision = class_precision.mean().item()
        val_recall = class_recall.mean().item()
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Training Loss: {train_loss:.4f},   Validation Loss: {val_loss:.4f}")
        print(f"Precision:     {val_precision:.2f},     Recall:          {val_recall:.2f}\n")

        # Save the best model
        if val_precision > best_val_precision:
            best_val_epoch = epoch
            best_val_precision = val_precision
            best_conf_matrix = conf_matrix
            torch.save(model.state_dict(), best_weight_path)

    print("Training completed!\n")
    end_time = time.time()  # End timer
    total_time = (end_time - start_time)/3600
    print(f"Total training time: {total_time:.2f} hours\n")

    print(f"Performance Summary (best model at epoch {best_val_epoch}):")
    print("-----------------------------")
    print("Class\tPrecision\tRecall")
    for i in range(num_classes):
        class_name = class_names[i]
        precision = class_precisions[best_val_epoch][i].item()
        recall = class_recalls[best_val_epoch][i].item()
        print(f"{class_name}\t{precision:.2f}\t\t{recall:.2f}")
    print("-----------------------------\n")

    # Plot and save figures
    plot_losses(train_losses, val_losses, num_epochs, save_folder)
    plot_metrics(val_recalls, val_precisions, num_epochs, save_folder)
    plot_confusion_matrix(best_conf_matrix, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to the directory containing images")
    parser.add_argument("--val_dir", type=str, help="Path to the directory containing validation images")
    parser.add_argument("--weights", type=str, default='', help="Path to pre-trained weights (leave blank to train from scratch)")
    parser.add_argument("--age_in_months", type=int, default=0, help="Age of the infant in months")
    parser.add_argument("--blur", action="store_true", help="Apply blur transformation")
    parser.add_argument("--contrast", action="store_true", help="Apply contrast transformation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save_folder", type=str, default="results", help="Folder name to save best weights and plots")

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        weights = args.weights,
        age_in_months=args.age_in_months,
        apply_blur=args.blur,
        apply_contrast=args.contrast,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_folder=args.save_folder
    )
