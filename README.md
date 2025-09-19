# Skin Cancer Detection using CNNs and ViTs
Source code for the paper "Skin Cancer Detection Using Convolutional Networks and Vision Transformers".
## Scripts
- `caformer.py`: Training script for the CAFormer model.
- `deit_distilled.py`: Training script for the DeiT-Distilled model.
- `efficientnetv2.py`: Training script for the EfficientNetV2 model.
- `ensemble_load.py`: Script to load the trained models, perform weighted-average ensembling, and evaluate the results.
## How to Run
1. Train the individual models by running the three training scripts.
2. Once the models are saved, run `ensemble_load.py` to get the final evaluation.
