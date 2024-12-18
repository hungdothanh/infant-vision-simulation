import argparse
import os
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataloader import InfantVisionDataset
from utils.plots import plot_images_with_predictions



def validate(test_dir, batch_size, num_images_to_predict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize dataset and dataloader
    test_dataset = InfantVisionDataset(image_dir=test_dir, age_in_months=1000, apply_blur=False, apply_contrast=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = ResNet50(num_classes=2).to(device)
    model.load_state_dict(torch.load("best_weights/best_model.pth"))
    model.eval()

    # Evaluate on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")

    # Select random images from the test set
    random_indices = random.sample(range(len(test_dataset)), num_images_to_predict)

    images = []
    labels = []
    predictions = []

    with torch.no_grad():
        for idx in random_indices:
            image, label = test_dataset[idx]
            images.append(image)
            labels.append(label)

            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())

    # Plot the images with their corresponding labels and predictions
    plot_images_with_predictions(images, labels, predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the model on the test set")
    parser.add_argument("test_dir", type=str, help="Directory of the test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num_img", type=int, default=5, help="Number of random images to predict and plot")
    args = parser.parse_args()

    validate(args.test_dir, args.batch_size, args.num_img)