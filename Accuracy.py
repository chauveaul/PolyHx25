import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.transforms.functional import to_pil_image

class WildfireDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)  # Using read_image instead of undefined decode_image()
        label = int(self.img_labels.iloc[idx, 1])  # Ensure label is an integer

        if self.transform:
            image = to_pil_image(image)
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Dataset setup
dataset = WildfireDataset(
    annotations_file=r"C:\Users\lufai\Downloads\wildfire_list2.csv",
    img_dir=r"C:\Users\lufai\Downloads\archive\train",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
)

# Splitting dataset into train and test
train_size = int(0.83 * len(dataset))  # 83% training
test_size = len(dataset) - train_size  # Remaining for testing
train_set, test_set = random_split(dataset, [train_size, test_size])

# Loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)  # No shuffle for evaluation

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("EmberAlert.pth", map_location=device, weights_only=False)
model.to(device)

# Accuracy evaluation function
def evaluate_model_accuracy(loader, model, device):
    """Evaluates model accuracy using a DataLoader."""
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(f"good: {num_correct}")
            print(f"total: {num_samples}")
            

            num_correct += (preds == targets).sum().item()
            num_samples += targets.size(0)

    accuracy = (num_correct / num_samples) * 100
    print(f"Model accuracy: {accuracy:.2f}%")

# Run evaluation
evaluate_model_accuracy(test_loader, model, device)
