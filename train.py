import os
import sys
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


def train(train_dir, val_dir, age_in_months, apply_blur, apply_contrast, weights, num_epochs, batch_size, lr, unfreeze_layer, resume_checkpoint, save_folder_name):
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

    # Define class
    num_classes = 2
    class_names = ["dog", "cat"]

    # CUSTOMIZED RESNET50 WITH INPUT PRE-TRAINED WEIGHT --------------------------
    # model = ResNet50(num_classes=num_classes).to(device)

    
    # Load model based on input arguments
    print(f"Done! Loading the model with modified prediction layer to solve {num_classes}-class classification task...\n")
    if weights == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        print("\nUsing pre-trained ResNet50 weights from ImageNet.\n")
    elif weights:
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(weights))
        print(f"Loaded weights from {weights}\n")
    else:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        print("Training from scratch.\n")

    # Freeze layers based on the unfreeze_layer input
    if unfreeze_layer:
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False

        layers_to_unfreeze = unfreeze_layer.split(",")  # Split by comma for multiple layers
        layers_to_unfreeze = [layer.strip() for layer in layers_to_unfreeze]  # Clean up spaces

        for layer_name in layers_to_unfreeze:
            if hasattr(model, layer_name):
                target_layer = getattr(model, layer_name)
                for param in target_layer.parameters():
                    param.requires_grad = True
                print(f"Layer '{layer_name}' is now trainable. The others are freezed.")
            else:
                raise ValueError(f"Layer '{layer_name}' not found in the model.")


    model = model.to(device)

    """ Uncomment to print model architecture summary """
    # print("Done! Summary of model architecture: ")
    # summary(model, input_size=(3, 224, 224), device=str(device))

    # Define loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create directory for saving best weights and plots
    save_folder = os.path.join("results", save_folder_name)
    counter = 0
    while os.path.exists(save_folder):
        counter += 1
        save_folder = os.path.join("results", f"{save_folder_name}_{counter}")
    os.makedirs(save_folder, exist_ok=True)
    best_ckpt_path = os.path.join(save_folder, "best_ckpt.pt")
    last_ckpt_path = os.path.join(save_folder, "last_ckpt.pt")
    best_val_loss = float('inf')
    best_conf_matrix = torch.zeros(num_classes, num_classes)
    best_epoch = 0
    start_epoch = 0

    # Resume training if checkpoint is provided
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch

    # Lists for logging and plotting
    train_losses, val_losses = [], []
    val_precisions, val_recalls = [], []
    class_precisions, class_recalls = [], []

    start_time = time.time()  # Start timer
    # Start training...
    print("\nStart training! \n")
    for epoch in range(start_epoch, num_epochs):
        print(f"EPOCH {epoch + 1}/{num_epochs}:")
        # Train phase
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc="Training", unit="batch") as train_loader_tqdm:
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = torch.tensor([], dtype=torch.long).to(device)
        all_preds = torch.tensor([], dtype=torch.long).to(device)

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch") as val_loader_tqdm:
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)

                    all_labels = torch.cat((all_labels, labels.cpu()))
                    all_preds = torch.cat((all_preds, pred.cpu()))
                    # val_loader_tqdm.set_postfix(loss=loss.item())

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        class_precision, class_recall, conf_matrix = precision_recall(all_preds, all_labels, num_classes)

        class_precisions.append(class_precision)
        class_recalls.append(class_recall)
        val_precision = class_precision.mean().item()
        val_recall = class_recall.mean().item()
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Training Loss: {train_loss:.4f},   Validation Loss: {val_loss:.4f}")
        print(f"Precision:     {val_precision:.2f},     Recall:          {val_recall:.2f}\n")

        # Save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, last_ckpt_path)
        if val_loss < best_val_loss or best_val_loss is None:
            best_epoch = epoch
            best_val_loss = val_loss
            best_conf_matrix = conf_matrix
            torch.save(model.state_dict(), best_ckpt_path)



    end_time = time.time()  # End timer
    total_time = (end_time - start_time)/3600
    print(f"Training completed! - Total training time: {total_time:.2f} hours\n")

    print(f"Performance Summary (best model at epoch {best_epoch}):")
    print("-----------------------------")
    print("Class\tPrecision\tRecall")
    for i in range(num_classes):
        class_name = class_names[i]
        precision = class_precisions[best_epoch][i].item()
        recall = class_recalls[best_epoch][i].item()
        print(f"{class_name}\t{precision:.2f}\t\t{recall:.2f}")
    print("-----------------------------\n")

    print(f"Saving results and plots to {save_folder}...\n")
    # Plot and save figures
    plot_losses(train_losses, val_losses, save_folder)
    plot_metrics(val_recalls, val_precisions, save_folder)
    plot_confusion_matrix(best_conf_matrix, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to the directory containing images")
    parser.add_argument("--val_dir", type=str, help="Path to the directory containing validation images")
    parser.add_argument("--age", type=int, default=0, help="Age of the infant in months")
    parser.add_argument("--blur", action="store_true", help="Apply blur transformation")
    parser.add_argument("--contrast", action="store_true", help="Apply contrast transformation")
    parser.add_argument("--weights", type=str, default='', 
                        help="Path to pre-trained weights (leave blank to train from scratch/resnet50 to use pretrained w)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--unfreeze", type=str, default='', 
                        help="Comma-separated layer names to unfreeze, e.g., 'fc,layer4' (leave blank or by defaults to train normally)")
    parser.add_argument("--resume", type=str, default='', help="Path to checkpoint to resume training")
    parser.add_argument("--name", type=str, default="exp", help="Folder name to save checkpoints and plots")

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        age_in_months=args.age,
        apply_blur=args.blur,
        apply_contrast=args.contrast,
        weights = args.weights,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_layer=args.unfreeze,
        resume_checkpoint=args.resume,
        save_folder_name=args.name
    )
