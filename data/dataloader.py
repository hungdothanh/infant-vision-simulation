import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import pandas as pd


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
        sigma = max(min_sigma, max_sigma - (self.age_in_months / 60) * (max_sigma - min_sigma))
        return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))

    def get_contrast_transform(self):
        age_in_weeks = self.age_in_months * 4.348125  # max age months = 125
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

        filename = os.path.basename(image_path)
        label = 0 if "dog" in filename else 1  # 0 for dog, 1 for cat

        if self.apply_blur and self.blur_transform:
            image = self.blur_transform(image)
        if self.apply_contrast and self.contrast_transform:
            image = self.contrast_transform(image)
        if self.transform:
            image = self.transform(image)

        return image, label



