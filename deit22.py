# %%
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from transformers import AutoImageProcessor, DeiTForImageClassification, TrainingArguments, Trainer, BeitForImageClassification, SwinForImageClassification, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
# import DeiT2
from datasets import load_dataset, load_metric
import evaluate
# %%
import os

class CustomImageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.classes = sorted(os.listdir(data_folder))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.load_dataset()
        self.transform = transform

    def load_dataset(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_folder, class_name)
            if os.path.isdir(class_path):
                class_idx = self.class_to_idx[class_name]
                for filename in os.listdir(class_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(class_path, filename)
                        images.append((image_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "label": label}

# %%
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset = CustomImageDataset(r"C:\Users\Batur\Desktop\CNN-TA\CNN-TA\resources3\test", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# %%
# image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# %%

training_args = TrainingArguments(
    output_dir='./results_of_deittt2',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
)

# %%
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,
    eval_dataset=val_dataset,            
)
# %%
trainer.train()

# %%

trainer.save_model("./results_of_deittt2")

# %%

test_data_dir = r"C:\Users\Batur\Desktop\CNN-TA\CNN-TA\resources3\test2"
test_dataset = CustomImageDataset(test_data_dir, transform=transform)

# %%
predictions, labels, metrics = trainer.predict(test_dataset)


loss = nn.CrossEntropyLoss()

# %%
def calculate_accuracy(predictions, labels):
    correct_predictions = 0

    for pred, true_label in zip(predictions, labels):
        # Assuming predictions and labels are lists of lists or arrays
        if pred == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(labels)
    return accuracy


labels_ground = [item["label"] for item in test_dataset]

a = torch.max(torch.tensor(predictions[:, :3]), dim=1).indices
b = a.tolist()

accuracy = calculate_accuracy(b, labels_ground)
# %%
with open(r"C:\Users\Batur\Desktop\CNN-TA\CNN-TA\deittttpredictiond.txt", "w") as file:
    file.write(str(b))
    file.write("\n")
    for p in predictions:
        file.write(str(p))
        file.write("\n")
    file.write("\n")
    file.write(str(metrics))
    for pred in predictions[:, :3]:
        file.write(str(pred))
        file.write("\n")



# %%
from sklearn.metrics import f1_score

#%%
f1 = f1_score(labels_ground, b, average='micro')

print(f1)
# %%
