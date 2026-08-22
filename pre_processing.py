from medmnist import PneumoniaMNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# =========================================================
# 1. TRAINING TRANSFORM
# =========================================================

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


# =========================================================
# 2. VALIDATION AND TEST TRANSFORM
# =========================================================

test_transform = transforms.Compose([
    transforms.ToTensor()
])


# =========================================================
# 3. LOAD TRAINING DATASET
# =========================================================

train_dataset = PneumoniaMNIST(
    split="train",
    transform=train_transform,
    download=True
)


# =========================================================
# 4. LOAD VALIDATION DATASET
# =========================================================

val_dataset = PneumoniaMNIST(
    split="val",
    transform=test_transform,
    download=True
)


# =========================================================
# 5. LOAD TEST DATASET
# =========================================================

test_dataset = PneumoniaMNIST(
    split="test",
    transform=test_transform,
    download=True
)


# =========================================================
# 6. DATASET INFORMATION
# =========================================================

print("Dataset Information")
print("-------------------")

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# =========================================================
# 7. CHECK ONE PROCESSED TRAINING IMAGE
# =========================================================

image, label = train_dataset[0]

print("\nProcessed Image Information")
print("---------------------------")

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())
print("Label:", label)


# =========================================================
# 8. CREATE DATALOADERS
# =========================================================

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =========================================================
# 9. CHECK ONE TRAINING BATCH
# =========================================================

images, labels = next(iter(train_loader))

print("\nTraining Batch Information")
print("--------------------------")

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# =========================================================
# 10. CHECK ONE VALIDATION BATCH
# =========================================================

val_images, val_labels = next(iter(val_loader))

print("\nValidation Batch Information")
print("----------------------------")

print("Images shape:", val_images.shape)
print("Labels shape:", val_labels.shape)


# =========================================================
# 11. CHECK ONE TEST BATCH
# =========================================================

test_images, test_labels = next(iter(test_loader))

print("\nTest Batch Information")
print("----------------------")

print("Images shape:", test_images.shape)
print("Labels shape:", test_labels.shape)


# =========================================================
# 12. DISPLAY PROCESSED IMAGE
# =========================================================

plt.imshow(images[0].squeeze(), cmap="gray")

plt.title(
    f"Processed X-Ray - Label: {labels[0].item()}"
)

plt.axis("off")
plt.show()


# =========================================================
# 13. FINAL CHECK
# =========================================================

print("\nPreprocessing pipeline completed successfully!")