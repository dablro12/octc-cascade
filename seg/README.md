# Catheter&Electrode Semantic Segmentation on Infatogram with [SNUH-MEDISC](https://snuh.medisc.org/)

## Project Overview
This repository contains the code for the catheter electrode segmentation project on infantograms. The project aims to develop and train deep learning models that can automatically segment catheter electrodes from chest X-ray images. This is particularly useful in medical image processing where precise detection of medical devices in radiographic images is required.

## Repository Structure
Below is the directory structure of the `chest-segmentation` project, which includes Python scripts, Jupyter notebooks, and various utilities for training and inference:

```
chest-segmentation
├── inference
│   ├── script
│   │   └── __init__.py
│   └── utils
│       └── __init__.py
├── README.md
├── settings
│   └── __init__.py
└── trainer
    ├── model
    │   ├── deb
    │   │   ├── swinunet_train.ipynb
    │   │   └── swinunet_train.py
    │   ├── load_model.py
    │   ├── manet.py
    │   ├── models.py
    │   ├── modules
    │   │   ├── conv.py
    │   │   ├── __init__.py
    │   ├── swinunet.py
    │   ├── unet_plus_plus.py
    │   └── unet.py
    ├── test
    │   ├── inference.py
    │   ├── __init__.py
    │   └── train.ipynb
    ├── train
    │   ├── __init__.py
    │   ├── run.py
    │   ├── run.sh
    │   ├── train.ipynb
    │   └── train.py
    └── utils
        ├── arg.py
        ├── custom_transforms.py
        ├── dataset.py
        ├── __init__.py
        ├── metrics.py
```

### Key Components:
- **inference/**: Scripts and utilities for model inference after training.
- **trainer/**: Contains all scripts and modules necessary for training models.
  - **model/**: Definitions and implementations of various segmentation models like U-Net, U-Net++, SwinUNet, and MAnet.
  - **train/**: Scripts and notebooks for running training processes.
  - **test/**: Scripts for evaluating the trained models and performing inference.
- **utils/**: Utility scripts including dataset handling, custom transformations, and performance metrics.
- **settings/**: Configuration files and settings for the project.

## Setup and Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dablro12/chest-segmentation.git
   cd chest-segmentation
   ```

2. Install required packages:
   ```bash
   pip install -r src/settings/requirements.txt
   ```

## Usage
To train a model, navigate to the `trainer/train/` directory and run the following command:
```bash
sh src/trainer/train/run.sh
```

To perform inference using a trained model, navigate to the `trainer/test/` directory and execute:
```bash
python inference.py
```

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your features or fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
