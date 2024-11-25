import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time


class InfantVisionDataset(Dataset):
    def __init__(self, image_dir, age_in_months, apply_blur, apply_contrast, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.age_in_months = age_in_months
        self.apply_blur = apply_blur
        self.apply_contrast = apply_contrast
        self.transform = transform
        self.blur_transform = self.get_acuity_blur_transform() if apply_blur else None
        self.contrast_transform = self.get_contrast_transform() if apply_contrast else None

    def get_acuity_blur_transform(self):
        kernel_size = 15
        max_sigma = 4.0
        min_sigma = 0.1
        sigma = max(min_sigma, max_sigma - (self.age_in_months / 12) * (max_sigma - min_sigma))
        return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))

    def get_contrast_transform(self):
        age_in_weeks = self.age_in_months * 4.348125
        contrast_factor = min(age_in_weeks / 500, 1)

        def my_adjust_contrast():
            def _func(img):
                return transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor)
            return _func

        return my_adjust_contrast()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.apply_blur and self.blur_transform:
            image = self.blur_transform(image)
        if self.apply_contrast and self.contrast_transform:
            image = self.contrast_transform(image)
        if self.transform:
            image = self.transform(image)

        return image


def measure_loading_performance(data_loader, total_images=100):
    start_time = time.time()
    images_loaded = 0
    for images in data_loader:
        images_loaded += images.size(0)
        if images_loaded >= total_images:
            break
    end_time = time.time()
    return end_time - start_time
