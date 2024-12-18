# infant-vision-simulation
 Simulates infant vision development by applying Gaussian blur and contrast adjustments to images based on age. Trains ResNet50 on these transformations to test their impact on classification performance.

## Install
Clone repo and install [requirements.txt](./requirements.txt) in a Python>=3.8.0 environment, including PyTorch>=1.8.

```
git clone https://github.com/hungdothanh/infant-vision-simulation.git  
cd infant-vision-simulation
pip install -r requirements.txt 
```

## Setup
Modify the correct directory path to the training set and validation set folder defined in the data/data.yaml

## Train
The following command executes training.
```
python train.py --data 'data/data.yaml' --age '0,30,60' --blur --epochs 10 --batch_size 64 --lr 0.01 --name 'cirriculum1-blur'
```
