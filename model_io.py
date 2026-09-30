import torch
import torch.nn as nn


print("PyTorch imported successfully!")

print(
    "PyTorch version:",
    torch.__version__
)

# ==========================================
# STEP 2 — CREATE CNN MODEL
# ==========================================

class PneumoniaCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            32 * 7 * 7,
            64
        )

        self.fc2 = nn.Linear(
            64,
            2
        )

        # Activation
        self.relu = nn.ReLU()


    def forward(self, x):

        # First convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(
            x.size(0),
            -1
        )

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)

        # Output
        x = self.fc2(x)

        return x


# ==========================================
# CREATE MODEL
# ==========================================

model = PneumoniaCNN()

print("\nCNN model created successfully!")

print(model)

# ==========================================
# STEP 3 — GET HOSPITAL DATA
# ==========================================

import numpy as np
from medmnist import PneumoniaMNIST
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor
)
from torch.utils.data import Dataset, DataLoader


# ==========================================
# MONAI TRANSFORM
# ==========================================

transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    ToTensor()
])


# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_dataset = PneumoniaMNIST(
    split="train",
    download=False
)


# ==========================================
# LOAD HOSPITAL PARTITIONS
# ==========================================

partitions = np.load(
    "hospital_partitions.npz"
)


# ==========================================
# CREATE DATASET CLASS
# ==========================================

class HospitalDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        transform
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = self.indices[index]

        image, label = self.base_dataset[real_index]

        image = np.array(image)

        image = self.transform(image)

        label = torch.tensor(
            int(label[0]),
            dtype=torch.long
        )

        return image, label


# ==========================================
# CREATE HOSPITAL 1 DATASET
# ==========================================

hospital_1_dataset = HospitalDataset(
    train_dataset,
    partitions["hospital_1"],
    transform
)


# ==========================================
# CREATE HOSPITAL 1 DATALOADER
# ==========================================

hospital_1_loader = DataLoader(
    hospital_1_dataset,
    batch_size=32,
    shuffle=True
)


# ==========================================
# GET ONE BATCH
# ==========================================

images, labels = next(
    iter(hospital_1_loader)
)


print("\n================================")
print("HOSPITAL 1 INPUT")
print("================================")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)

# ==========================================
# STEP 4 — MODEL INPUT / OUTPUT TEST
# ==========================================

model.eval()

with torch.no_grad():

    outputs = model(images)


print("\n================================")
print("MODEL OUTPUT TEST")
print("================================")

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Output dtype:", outputs.dtype)

# ==========================================
# STEP 5 — GENERATE PREDICTIONS
# ==========================================

predictions = torch.argmax(
    outputs,
    dim=1
)

print("\n================================")
print("PREDICTION TEST")
print("================================")

print("Predictions shape:", predictions.shape)

print(
    "First 10 predictions:",
    predictions[:10].tolist()
)

print(
    "First 10 actual labels:",
    labels[:10].tolist()
)

# ==========================================
# STEP 6 — CALCULATE MODEL LOSS
# ==========================================

criterion = nn.CrossEntropyLoss()

loss = criterion(
    outputs,
    labels
)

print("\n================================")
print("LOSS CALCULATION TEST")
print("================================")

print("Model output shape:", outputs.shape)
print("Labels shape:", labels.shape)
print("Loss:", loss.item())

# ==========================================
# STEP 7 — FINAL MODEL I/O VERIFICATION
# ==========================================

print("\n========================================")
print("DELIVERABLE 6 FINAL VERIFICATION")
print("========================================")

success = True


# ------------------------------------------
# 1. Check input
# ------------------------------------------

if images.shape != (32, 1, 28, 28):
    success = False

print("Input shape:", images.shape)


# ------------------------------------------
# 2. Check model output
# ------------------------------------------

if outputs.shape != (32, 2):
    success = False

print("Output shape:", outputs.shape)


# ------------------------------------------
# 3. Check predictions
# ------------------------------------------

if predictions.shape != (32,):
    success = False

print("Prediction shape:", predictions.shape)


# ------------------------------------------
# 4. Check loss
# ------------------------------------------

if not torch.isfinite(loss):
    success = False

print("Loss:", loss.item())


# ------------------------------------------
# 5. Final result
# ------------------------------------------

if success:

    print("\n========================================")
    print("DELIVERABLE 6: COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        "Model input, output, prediction, "
        "and loss handling are working correctly."
    )

else:

    print("\n========================================")
    print("DELIVERABLE 6: VERIFICATION FAILED")
    print("========================================")