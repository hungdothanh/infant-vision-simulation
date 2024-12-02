
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataloader import InfantVisionDataset
from model import ResNet50


def validate(test_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize dataset and dataloader
    test_dataset = InfantVisionDataset(image_dir=test_dir, transform=transform)
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

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, help="Path to the test data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    validate(
        test_dir=args.test_dir,
        batch_size=args.batch_size
    )
