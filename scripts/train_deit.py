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
print(f"Usando o dispositivo: {device}")
nome_do_modelo = 'deit_base_distilled_patch16_224'
base_utilizada = 'HAM 10000'

base_dir = ""

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


def save_model(model, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(
        save_dir, f"model_{nome_do_modelo}_base_{base_utilizada}_batch_{batch_size}.pth")
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


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


train_dir = os.path.join(base_dir, 'training')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


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

model = timm.create_model(
    nome_do_modelo, pretrained=True, num_classes=len(class_names))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

patience = 10
best_val_acc = 0.0
early_stopping_counter = 0
best_model_wts = None


def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=30):
    global best_val_acc, early_stopping_counter, best_model_wts

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(
            f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = model.state_dict()
            early_stopping_counter = 0
            save_model(model)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    if best_model_wts:
        model.load_state_dict(best_model_wts)
        print("Best model loaded with validation accuracy:", best_val_acc)
    return model


best_model = train_model(model, train_loader, val_loader,
                         optimizer, loss_fn, epochs=100)


best_model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(
    f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {100 * correct / total:.2f}%')


y_pred = []
y_true = []

best_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

print('Classification Report')
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
