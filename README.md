# Skin Cancer Detection using CNNs and Vision Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the paper: **"Skin Cancer Detection Using Convolutional Networks and Vision Transformers"**.

**Authors:** Davi Israel Abtibol Carvalho, Fábio Cavalcante Binatti, Marly. G. F. Costa, Cícero F.F. Costa Filho

DOI: **https://doi.org/10.1109/SIPAIM67325.2025.11283391**

## Abstract

Skin cancer, affecting over 1.5 million in 2022, poses a significant public health concern... An accuracy of 93.61% and an F1-score of 96.04% was obtained.

## Getting Started

### Prerequisites

*   Python 3.12.1
*   PyTorch
*   CUDA

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbtibolDavi/skin-cancer-ensemble-paper.git
    cd skin-cancer-ensemble-paper
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset:**
    Download the HAM10000 dataset from the Harvard Dataverse:
    - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

5. **Place it in the `data/` directory**

## Usage

1.  **Train the individual models:**
    ```bash
    python scripts/train_caformer.py
    python scripts/train_deit.py
    python scripts/train_efficientnet.py
    ```
2.  **Run the ensemble evaluation:**
    ```bash
    python scripts/ensemble.py
    ```

## Results

Our ensemble model, combining EfficientNetV2, DeiT Distilled, and CAFormer, achieved the following results on the HAM10000 dataset for binary classification:

| Model          | Accuracy (%) | Balanced Accuracy (%) | F1-Score (%) |
| -------------- | ------------ | --------------------- | ------------ |
| **Ensemble**   | **93.61**    | **89.42**             | **96.04**    |
| EfficientNetV2 | 92.81        | 89.50                 | 95.62        |

<p float="left">
  <img src="confusion_matrix_ensemble_image.png" width="49%" />
  <img src="confusion_matrix_efficientnet_image.png" width="49%" />
</p>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was funded by Samsung Electronics of Amazonia Ltda., under the terms of Federal Law n°8.387/1991, agreement 001/2020, signed with UFAM/FAEPI, Brazil.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11283391,
  author={Carvalho, Davi Israel Abtibol and Cavalcante Binatti, Fábio and Costa, Marly. G. F. and Filho, Cícero F.F. Costa},
  booktitle={2025 21st International Symposium on Biomedical Image Processing and Analysis (SIPAIM)}, 
  title={Skin Cancer Detection Using Convolutional Networks and Vision Transformers}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Deep learning;Solid modeling;Computer vision;Visualization;Accuracy;Computational modeling;Transformers;Skin;Lesions;Skin cancer;skin cancer detection;ensemble methods;machine learning;CNNs;ViTs;binary classification},
  doi={10.1109/SIPAIM67325.2025.11283391}}
