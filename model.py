import numpy as np
from matplotlib import pyplot as plt
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, swin_v2_t
from tqdm import tqdm

ds = load_dataset(
    "mikkoim/aquamonitor-jyu",
    cache_dir="."
)

hf_hub_download(
    repo_id="mikkoim/aquamonitor-jyu",
    filename="aquamonitor-jyu.parquet.gzip",
    repo_type="dataset",
    local_dir="."
)

# The keys match the rows in the metadata table
metadata = pd.read_parquet("./aquamonitor-jyu.parquet.gzip")

classes = sorted(metadata["taxon_group"].unique())
class_map = {k:v for v,k in enumerate(classes)}
class_map_inv = {v:k for k,v in class_map.items()}

metadata["img"] = metadata["img"].str.removesuffix(".jpg")
label_dict = dict(zip(metadata["img"], metadata["taxon_group"].map(class_map)))

IMAGE_SIZE = 224
BATCH_SIZE = 16

tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess(batch):
    return {"key": batch["__key__"],
            "img": [tf(x) for x in batch["jpg"]],
            "label": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.long)}


eval_ds = ds["validation"].with_transform(preprocess)

print(f"Development Size: {eval_ds.num_rows}")

eval_loader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE
)

class EnsembleModel(nn.Module):
    def __init__(self, class_num):
        super(EnsembleModel, self).__init__()

        resnet = resnet18()
        resnet.fc = nn.Linear(resnet.fc.in_features, class_num)

        swin = swin_v2_t()
        swin.head = nn.Linear(swin.head.in_features, class_num)

        self.resnet = resnet
        self.swin = swin

    def forward(self, x):

        resnet_out = self.resnet(x)
        swin_out = self.swin(x)

        out = (resnet_out + swin_out) / 2.0
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dict = torch.load(
    "./model.pt",
    weights_only=True,
    map_location=device
)

model = EnsembleModel(len(classes))
model.resnet.load_state_dict(weight_dict["resnet"])
model.swin.load_state_dict(weight_dict["swin"])

model.to(device)
model.eval()

eval_labels = []
eval_preds = []

with torch.no_grad():
    for batch in tqdm(eval_loader):
        images, labels = batch["img"], batch["label"]
        images, labels = images.to(device), labels.to(device)

        out = model(images)

        # Labels und Predictions sammeln
        _, preds = torch.max(out.data, 1)
        eval_labels.extend(labels.cpu().numpy())
        eval_preds.extend(preds.cpu().numpy())

eval_f1 = f1_score(eval_labels, eval_preds, average="weighted")
print(f"F1-Score: {eval_f1:.3f}")

# 1) Confusion Matrix berechnen
cm = confusion_matrix(eval_labels, eval_preds)

# 2) (Optional) Normalisieren, um relative Häufigkeiten anzuzeigen
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# 3) Plotten
plt.figure(figsize=(12,12))
plt.imshow(cm_normalized, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

# Achsenbeschriftungen
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.tight_layout()
plt.show()