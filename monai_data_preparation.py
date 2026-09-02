import numpy as np
import torch

from medmnist import PneumoniaMNIST

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor
)

from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

train_dataset = PneumoniaMNIST(
    split="train",
    download=False
)

print("Training dataset loaded!")
print("Training samples:", len(train_dataset))


# ==========================================
# 2. LOAD VALIDATION DATA
# ==========================================

val_dataset = PneumoniaMNIST(
    split="val",
    download=False
)

print("Validation dataset loaded!")
print("Validation samples:", len(val_dataset))


# ==========================================
# 3. LOAD HOSPITAL PARTITIONS
# ==========================================

partitions = np.load(
    "hospital_partitions.npz"
)

print("\nHospital partitions loaded!")

for i in range(1, 6):

    print(
        f"Hospital {i}: "
        f"{len(partitions[f'hospital_{i}'])} samples"
    )

# ==========================================
# 4. CREATE TRAINING TRANSFORMS
# ==========================================

train_transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    ToTensor()
])


# ==========================================
# 5. CREATE VALIDATION TRANSFORMS
# ==========================================

val_transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    ToTensor()
])


print("\nMONAI transforms created successfully!")
print("Training transform: Ready")
print("Validation transform: Ready")

# ==========================================
# 6. CREATE PYTORCH DATASET CLASS
# ==========================================

class MedicalHospitalDataset(Dataset):

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

    def __getitem__(self, position):

        # Get original dataset index
        real_index = self.indices[position]

        # Get image and label
        image, label = self.base_dataset[real_index]

        # Convert PIL image to NumPy
        image = np.array(image)

        # Apply MONAI transforms
        image = self.transform(image)

        # Convert label to PyTorch tensor
        label = torch.tensor(
            int(label[0]),
            dtype=torch.long
        )

        return image, label


print("\nPyTorch medical dataset class created successfully!")

# ==========================================
# 7. CREATE HOSPITAL DATASETS
# ==========================================

hospital_datasets = {}

for i in range(1, 6):

    hospital_name = f"hospital_{i}"

    hospital_indices = partitions[
        hospital_name
    ]

    hospital_dataset = MedicalHospitalDataset(
        base_dataset=train_dataset,
        indices=hospital_indices,
        transform=train_transform
    )

    hospital_datasets[
        hospital_name
    ] = hospital_dataset

    print(
        f"{hospital_name} dataset created: "
        f"{len(hospital_dataset)} samples"
    )

# ==========================================
# 8. CREATE LOCAL DATALOADERS
# ==========================================

batch_size = 32

hospital_loaders = {}

for hospital_name, hospital_dataset in hospital_datasets.items():

    loader = DataLoader(
        hospital_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    hospital_loaders[hospital_name] = loader

    print(
        f"{hospital_name} DataLoader created"
    )

# ==========================================
# 10. CREATE VALIDATION DATASET
# ==========================================

validation_dataset = MedicalHospitalDataset(
    base_dataset=val_dataset,
    indices=np.arange(len(val_dataset)),
    transform=val_transform
)

print("\nValidation dataset prepared!")
print(
    "Validation samples:",
    len(validation_dataset)
)


# ==========================================
# 11. CREATE VALIDATION DATALOADER
# ==========================================

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

print("Validation DataLoader created!")


# ==========================================
# 12. TEST VALIDATION BATCH
# ==========================================

validation_images, validation_labels = next(
    iter(validation_loader)
)

print("\n================================")
print("VALIDATION BATCH TEST")
print("================================")

print(
    "Images shape:",
    validation_images.shape
)

print(
    "Labels shape:",
    validation_labels.shape
)

print(
    "Image dtype:",
    validation_images.dtype
)

print(
    "Label dtype:",
    validation_labels.dtype
)

# ==========================================
# 13. FINAL MODEL-READINESS VERIFICATION
# ==========================================

print("\n========================================")
print("DELIVERABLE 5 FINAL VERIFICATION")
print("========================================")

success = True


# ------------------------------------------
# Check all 5 hospital DataLoaders
# ------------------------------------------

for hospital_name, loader in hospital_loaders.items():

    images, labels = next(iter(loader))

    print(f"\n{hospital_name}")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    # Check image shape
    if images.shape != (32, 1, 28, 28):
        success = False

    # Check image data type
    if images.dtype != torch.float32:
        success = False

    # Check label data type
    if labels.dtype != torch.long:
        success = False


# ------------------------------------------
# Check validation DataLoader
# ------------------------------------------

validation_images, validation_labels = next(
    iter(validation_loader)
)

print("\nValidation")
print("Image shape:", validation_images.shape)
print("Label shape:", validation_labels.shape)
print("Image dtype:", validation_images.dtype)
print("Label dtype:", validation_labels.dtype)


if validation_images.shape != (32, 1, 28, 28):
    success = False

if validation_images.dtype != torch.float32:
    success = False

if validation_labels.dtype != torch.long:
    success = False


# ------------------------------------------
# Final result
# ------------------------------------------

if success:

    print("\n========================================")
    print("DELIVERABLE 5: COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        "All 5 hospitals and validation data "
        "are ready for PyTorch model training."
    )

else:

    print("\n========================================")
    print("DELIVERABLE 5: VERIFICATION FAILED")
    print("========================================")