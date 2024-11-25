import os
from data.dataloader import InfantVisionDataset, measure_loading_performance
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def visualize_images_in_grid(image_dir, ages, apply_blur, apply_contrast, num_images=5):
    fig, axes = plt.subplots(num_images, len(ages), figsize=(15, 20))

    all_images = [Image.open(os.path.join(image_dir, img)).convert("RGB") for img in os.listdir(image_dir)]

    for i in range(num_images):
        for j, age in enumerate(ages):
            dataset = InfantVisionDataset(image_dir=image_dir, age_in_months=age, apply_blur=apply_blur,
                                          apply_contrast=apply_contrast, transform=base_transform)
            image_transformed = dataset[i]
            axes[i, j].imshow(image_transformed.permute(1, 2, 0))
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Image {i + 1}, Age: {age} months')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Directory with images
    image_dir = 'data2'

    # enable for contrast
    ages = [24, 48, 150]

    # enable for blur
    #ages = [0, 6, 13]

    age_in_months = 6

    base_transform = transforms.ToTensor()

    ####################################################################################################################

    # Visualize images in a grid format for specified ages
    # enable for blur
    #visualize_images_in_grid(image_dir=image_dir, ages=ages, apply_blur=True, apply_contrast=False, num_images=5)

    # enable for contrast
    visualize_images_in_grid(image_dir=image_dir, ages=ages, apply_blur=False, apply_contrast=True, num_images=5)

    ####################################################################################################################
    # Plain dataset (without transformations)
    plain_dataset = InfantVisionDataset(image_dir=image_dir, age_in_months=age_in_months, apply_blur=False,
                                        apply_contrast=False, transform=base_transform)
    plain_loader = DataLoader(plain_dataset, batch_size=10, shuffle=False)

    # Measure loading time for plain dataset
    plain_time = measure_loading_performance(plain_loader, total_images=100)
    print(f"Loading time without transformation (plain dataset): {plain_time:.2f} seconds")

    ####################################################################################################################

    # Transformed dataset (Blur adjustment)
    transformed_dataset_blur = InfantVisionDataset(image_dir=image_dir, age_in_months=age_in_months, apply_blur=True,
                                                   apply_contrast=False, transform=base_transform)

    transformed_loader_blur = DataLoader(transformed_dataset_blur, batch_size=10, shuffle=False)

    # Measure loading time for 100 images with blur transformation
    blur_time = measure_loading_performance(transformed_loader_blur, total_images=100)
    print(f"Loading time with transformation (blur transformed dataset): {blur_time:.2f} seconds")

    ####################################################################################################################

    # Transformed dataset (contrast adjustment)
    transformed_dataset_contrast = InfantVisionDataset(image_dir=image_dir, age_in_months=age_in_months,apply_blur=False,
                                                       apply_contrast=True, transform=base_transform)

    transformed_loader_contrast = DataLoader(transformed_dataset_contrast, batch_size=10, shuffle=False)

    # Measure loading time for contrast transformed dataset
    contrast_time = measure_loading_performance(transformed_loader_contrast, total_images=100)
    print(f"Loading time with transformation (contrast transformed dataset): {contrast_time:.2f} seconds")

    ####################################################################################################################

