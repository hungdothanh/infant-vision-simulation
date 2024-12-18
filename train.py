

import os
import time
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import models, transforms
from torchsummary import summary

from data.dataloader import InfantVisionDataset
from tqdm import tqdm
from utils.plots import plot_losses, plot_metrics, plot_confusion_matrix, visualize
from utils.metrics import precision_recall

from torch.utils.tensorboard import SummaryWriter


def train(data, age_in_months, apply_blur, apply_contrast, weights, num_epochs, batch_size, lr, unfreeze_layer, resume_checkpoint, save_folder_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if CUDA is available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Device Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**2):.2f} MB")
    else:
        print("No CUDA devices available. Using CPU.\n")

    # Define directories from data.yaml
    with open(data, 'r') as file:
        data = yaml.safe_load(file)
    train_dir = data['train']
    val_dir = data['val']

    # Define class
    num_classes = data['num_classes']
    class_names = data['class_names']

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Training or validation directory not found. Please check the paths in data.yaml")
    
    # Define base transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders based on age_in_months
    train_loaders = []
    stage_boundaries = []
    # print number of images in the training dataset/dataloader, for each case if age_in_months is provided (this one all dataset used are the same, just different in transformation) and if not
    if age_in_months:
        age_in_months = age_in_months.split(",")
        age_in_months = [int(age) for age in age_in_months]
        print(f"Creating {len(age_in_months)} datasets and dataloaders for training w.r.t {len(age_in_months)} stages of cirriculum learning...\n")

        for i, age in enumerate(age_in_months):
            dataset = InfantVisionDataset(
                image_dir=train_dir,
                age_in_months=age,
                apply_blur=apply_blur,
                apply_contrast=apply_contrast,
                transform=transform
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_loaders.append(dataloader)
            stage_boundaries.append(num_epochs * (i + 1))
        print(f"Done. Number of images in the training set: {len(dataset)}\n")
    else:
        # if no age provided, set it to default 200 months
        print("Creating training dataset and dataloader...\n")
        age_in_months = 200
        dataset = InfantVisionDataset(
            image_dir=train_dir,
            age_in_months=age_in_months,
            apply_blur=apply_blur,
            apply_contrast=apply_contrast,
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(dataloader)
        print(f"Done. Number of images in the training set: {len(dataset)}\n")

    print("Creating validation dataset and dataloader...\n")
    val_dataset = InfantVisionDataset(
        image_dir=val_dir,
        age_in_months=age_in_months,
        apply_blur=False,
        apply_contrast=False,
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Done. Number of images in the validation set: {len(val_dataset)}\n")

    
    # Load model based on input arguments --- 'ResNet50'
    print(f"Done! Loading the model with modified prediction layer to solve {num_classes}-class classification task...\n")
    # Load pre-trained weights from ImageNet if weights is set to 'resnet50'
    if weights == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        print("\nUsing pre-trained ResNet50 weights from ImageNet.\n")
    # Load saved weights if provided
    elif weights:
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(weights))
        print(f"Loaded weights from {weights}\n")
    # Train from scratch
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

    # Save visualization for dataloaders with different transformations if apply_blur or apply_contrast is True
    if apply_blur or apply_contrast:
        transform_type = "Blur + Contrast" if apply_blur and apply_contrast else "Blur" if apply_blur else "Contrast" if apply_contrast else "None"
        num_images = 5
        print("Saving visualization of transformations for multi-dataloaders...\n")
        visualize(train_loaders, age_in_months, transform_type, num_images, save_folder)
        print(f"Done! Check {save_folder} for the visualization of transformations.\n")


    # Define paths for saving best and last checkpoints
    best_ckpt_path = os.path.join(save_folder, "best_ckpt.pt")
    last_ckpt_path = os.path.join(save_folder, "last_ckpt.pt")

    # Initialize variables for tracking best model
    best_val_loss = float('inf')
    best_conf_matrix = torch.zeros(num_classes, num_classes)
    best_epoch = 0
    start_epoch = 0

    # Lists for logging and plotting
    train_losses, val_losses = [], []
    val_precisions, val_recalls = [], []
    class_precisions, class_recalls = [], []


    # Resume training if checkpoint is provided
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch


    # Start timer
    start_time = time.time()  

    # Start training...
    print("\nStart training! \n")
    for stage_idx, train_loader in enumerate(train_loaders):
        if len(train_loaders) > 1:
            print(f"Stage {stage_idx + 1} - Age months {age_in_months[stage_idx]}:\n")
        for epoch in range(start_epoch, num_epochs * (stage_idx+1)):
            print(f"EPOCH {epoch}/{num_epochs * len(train_loaders)-1}")
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

                        all_labels = torch.cat((all_labels, labels))
                        all_preds = torch.cat((all_preds, pred))
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

            # Save last checkpoint for resuming training if needed
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),}, last_ckpt_path)
            
            # Save best checkpoint accross all stages 
            if val_loss < best_val_loss or best_val_loss is None:
                best_epoch = epoch
                best_val_loss = val_loss
                best_conf_matrix = conf_matrix
                torch.save(model.state_dict(), best_ckpt_path)

        start_epoch += num_epochs   

    # End timer
    end_time = time.time() 
    total_time = (end_time - start_time) / 3600
    print(f"Training completed! - Total training time: {total_time:.2f} hours\n")

    # Display performance summary (Best Precision and Recall for each class)
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
    plot_losses(train_losses, val_losses, save_folder, stage_boundaries)
    plot_metrics(val_precisions, val_recalls, save_folder, stage_boundaries)
    plot_confusion_matrix(best_conf_matrix, save_folder)

    print("Done! Checkpoints and plots saved successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the directory containing images")
    parser.add_argument("--age", type=str, default='', help="Comma-separated list of ages in months for each stage of cirriculum learning")
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
        data=args.data,
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

# Run the following command to train the model:
# python train.py --data 'data/data.yaml' --age '0,30,60' --blur --epochs 10 --batch_size 64 --lr 0.01 --name 'cirriculum1-blur'