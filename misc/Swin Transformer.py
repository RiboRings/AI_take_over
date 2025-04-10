import torch

# Check and set CUDA if available
print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -*- coding: utf-8 -*-
"""aquamonitor-jyu-cpu

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Fas7M-qiIbq0D2aSd22Th6fiYWstR-TN
"""

#!pip install datasets

from datetime import datetime
t = datetime.now()

from datasets import load_dataset
ds = load_dataset("mikkoim/aquamonitor-jyu", cache_dir="Dataset/data")

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="mikkoim/aquamonitor-jyu", filename="aquamonitor-jyu.parquet.gzip", repo_type="dataset", local_dir=".")

# dataset elements can be accessed with indices. Each "row" or record
# has an image and a key that can be used to access data from the metadata table
record = ds["train"][0]
record

img = record["jpg"]
print(record["__key__"])
img

import pandas as pd

# The keys match the rows in the metadata table
metadata = pd.read_parquet('aquamonitor-jyu.parquet.gzip')
metadata

# The class map is used to map the label string representations to integers, like torch expects

classes = sorted(metadata["taxon_group"].unique())
class_map = {k:v for v,k in enumerate(classes)}
class_map_inv = {v:k for k,v in class_map.items()}
class_map_inv

# We define a label dict that is used to map the image key (filename) into a class
metadata["img"] = metadata["img"].str.removesuffix(".jpg")
label_dict = dict(zip(metadata["img"], metadata["taxon_group"].map(class_map)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# This preprocessing function is applied over all samples in the HuggingFace dataset
# the transform is applied to each image
# the label is defined by using the key
def preprocess(batch):
    return {"key": batch["__key__"],
            "img": [tf(x) for x in batch["jpg"]],
            "label": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.long)}

ds_train = ds["train"].with_transform(preprocess)
ds_val = ds["validation"].with_transform(preprocess)

# Dataloader definition
batch_size = 32
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=batch_size)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For Jupyter tqdm.notebook
from sklearn.metrics import f1_score
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


def validate(model, val_dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Validation F1 Score: {f1:.4f}")

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Loss: {loss.item()}")

        train_losses.append(total_loss/len(train_dataloader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}")
        validate(model, val_dataloader)

print(datetime.now() - t)

model = swin_v2_b(weights = Swin_V2_B_Weights.DEFAULT)
in_features = model.head.in_features
model.head = nn.Linear(in_features, len(classes))
model.to(device)

print(f"Modell wird auf folgendem Device ausgeführt: {next(model.parameters()).device}")
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
val_losses = []
num_epochs = 20

train(model=model,
      train_dataloader=dl_train,
      val_dataloader=dl_val,
      criterion=criterion,
      optimizer=optimizer,
      epochs=num_epochs)

print(datetime.now() - t)

# Saving the model checkpoint
torch.save(model.state_dict(), "model.pt")

# The submission should include a model.py that includes the model definition:
class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, n_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
# And the model file 'model5epochs.pt' that should be possible to load with torch

model_eval = swin_v2_b()
in_features = model_eval.head.in_features
model_eval.head = nn.Linear(in_features, len(classes))
model_eval.load_state_dict(torch.load("model5epochs.pt", weights_only=True))
model_eval.to(device)
print(f"Modell wird auf folgendem Device ausgeführt: {next(model_eval.parameters()).device}")

model_eval.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    total_loss = 0
    for batch in tqdm(dl_val):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        outputs = model_eval(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        loss = criterion(outputs, labels)
        total_loss += loss.item()

    val_losses.append(total_loss / len(dl_val))

y_pred = [class_map_inv[x] for x in all_preds]
y = [class_map_inv[x] for x in all_labels]

from sklearn.metrics import classification_report
print(classification_report(y,
                            y_pred,
                            zero_division=0))

print(datetime.now() - t)
print(train_losses)
print(val_losses)

import matplotlib.pyplot as plt
# Plot erstellen
plt.figure(figsize=(8,6))
plt.plot(range(num_epochs), train_losses, label="Train Loss", color="blue")

# Achsen und Titel setzen
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()  # Legende anzeigen
plt.grid(True)  # Hilfslinien für bessere Lesbarkeit

# Graph anzeigen
plt.show()
