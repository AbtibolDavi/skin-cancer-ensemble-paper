from sklearn.metrics import balanced_accuracy_score
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import timm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = ""
train_dir = os.path.join(base_dir, 'training')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
batch_size = 32
img_size = (224, 224)

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.Flip(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
    A.RandomBrightnessContrast(),
    A.HueSaturationValue(),
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.GaussianBlur(blur_limit=(3, 7)),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_test_transforms = A.Compose([
    A.Resize(226, 226),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class AlbumentationsDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, index):
        image_path, label = self.image_folder_dataset.samples[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label


image_train_ds = datasets.ImageFolder(train_dir)
image_val_ds = datasets.ImageFolder(val_dir)
image_test_ds = datasets.ImageFolder(test_dir)

train_ds = AlbumentationsDataset(image_train_ds, transform=train_transforms)
val_ds = AlbumentationsDataset(image_val_ds, transform=val_test_transforms)
test_ds = AlbumentationsDataset(image_test_ds, transform=val_test_transforms)

class_names = train_ds.image_folder_dataset.classes
print("Class Names:", class_names)

train_targets = np.array(train_ds.image_folder_dataset.targets)
class_counts = np.bincount(train_targets)
class_weights = 1. / class_counts
samples_weights = [class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(
    samples_weights, num_samples=len(samples_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

model_paths = [
    r"saved_models\modelo_caformer_m36.sail_in22k_ft_in1k_base_HAM 10000 Davi_lote_32.pth",
    r"saved_models\modelo_deit_base_distilled_patch16_224_base_HAM 10000 Davi_lote_32.pth",
    r"saved_models\modelo_gc_efficientnetv2_rw_t.agc_in1k_base_HAM 10000 Davi_lote_32.pth"
]

model_names = [
    'caformer_m36.sail_in22k_ft_in1k',
    'deit_base_distilled_patch16_224',
    'gc_efficientnetv2_rw_t.agc_in1k'
]

models = []
for path, name in zip(model_paths, model_names):
    model = timm.create_model(name, pretrained=False,
                              num_classes=len(class_names))
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    models.append(model)


def get_predictions(model, data_loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_pred, y_true


def calculate_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


val_preds = []
for model in models:
    y_pred, y_true = get_predictions(model, val_loader)
    val_preds.append((y_pred, y_true))

accuracies = []
for y_pred, y_true in val_preds:
    accuracy = calculate_accuracy(y_true, y_pred)
    accuracies.append(accuracy)

total_accuracy = sum(accuracies)
weights = [acc / total_accuracy for acc in accuracies]

print("Model Accuracies:", accuracies)
print("Weights:", weights)

all_preds = []
for model in models:
    y_pred, y_true = get_predictions(model, test_loader)
    all_preds.append(y_pred)

final_preds = np.zeros((len(y_true), len(class_names)),
                       dtype=np.float32)
for i, preds in enumerate(all_preds):
    preds_one_hot = np.eye(len(class_names))[preds]
    final_preds += weights[i] * preds_one_hot

final_preds = np.argmax(final_preds, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_true, final_preds))

print('Classification Report')
print(classification_report(y_true, final_preds, target_names=class_names))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, final_preds), annot=True,
            fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

all_balanced_acc = []

for model in models:
    y_pred, y_true = get_predictions(model, test_loader)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    all_balanced_acc.append(balanced_acc)

ensemble_balanced_acc = balanced_accuracy_score(y_true, final_preds)
print("Balanced Accuracy of Individual Models:", all_balanced_acc)
print("Balanced Ensemble Accuracy:", ensemble_balanced_acc)

test_loader_no_shuffle = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False)

print("\n--- Measuring Inference Time ---")
starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)

single_model_to_test = models[2]
single_model_to_test.eval()

for _ in range(5):
    _ = single_model_to_test(torch.randn(batch_size, 3, 224, 224).to(device))

with torch.no_grad():
    single_model_times = []
    for images, _ in test_loader_no_shuffle:
        images = images.to(device)
        starter.record()
        _ = single_model_to_test(images)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        single_model_times.append(curr_time)
avg_single_model_time_per_batch = np.mean(single_model_times)
print(
    f"Individual Model (EfficientNetV2): Average time per batch: {avg_single_model_time_per_batch:.2f} ms")

with torch.no_grad():
    ensemble_times = []
    for images, _ in test_loader_no_shuffle:
        images = images.to(device)
        starter.record()
        for model in models:
            _ = model(images)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        ensemble_times.append(curr_time)
avg_ensemble_time_per_batch = np.mean(ensemble_times)
print(
    f"Ensemble (3 models): Average time per batch: {avg_ensemble_time_per_batch:.2f} ms")

print("\n--- Measuring Peak VRAM Memory Usage ---")
torch.cuda.reset_peak_memory_stats(device)
dummy_batch = torch.randn(batch_size, 3, 224, 224).to(device)

with torch.no_grad():
    _ = single_model_to_test(dummy_batch)
peak_mem_single = torch.cuda.max_memory_allocated(device) / 1024**2
print(
    f"Individual Model (EfficientNetV2): Peak VRAM per batch: {peak_mem_single:.2f} MB")

torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
    for model in models:
        _ = model(dummy_batch)
peak_mem_ensemble_inference = torch.cuda.max_memory_allocated(device) / 1024**2
print(
    f"Ensemble (inference of 1 batch): VRAM peak during inference: {peak_mem_ensemble_inference:.2f} MB")
