
"""
In this project, the Cat vs Dog dataset from Kaggle is used to train and validate the model.
The script downloads the dataset from the google drive link provided below.
The dataset is saved in a folder named 'dataset' in the root directory.
The dataset have the following structure:
- catdog
    - train
        - image1
        - image2
        - ...
    - val
        - image1
        - image2
        - ...
    - data.yaml
"""

import gdown

url = 'https://drive.google.com/drive/folders/1P3VEz-pNoKzoAxVxDCLU5EETiFUXqVBP?usp=drive_link'
output = 'dataset'
gdown.download(url, output, quiet=False)

