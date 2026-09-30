import numpy as np
import torch
import matplotlib.pyplot as plt

from medmnist import PneumoniaMNIST

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor
)

from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. MONAI TRANSFORM PIPELINE
# ==========================================

medical_transform = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    ToTensor()
])


# ==========================================
# 2. LOAD MEDICAL DATASET
# ==========================================

train_dataset = PneumoniaMNIST(
    split="train",
    download=False
)

print("Medical dataset loaded!")
print("Total training samples:", len(train_dataset))


# ==========================================
# 3. LOAD HOSPITAL PARTITIONS
# ==========================================

partitions = np.load(
    "hospital_partitions.npz"
)

print("\nHospital partitions loaded!")


# ==========================================
# 4. CREATE HOSPITAL DATASET CLASS
# ==========================================

class HospitalDataset(Dataset):

    def __init__(self, base_dataset, indices, transform):

        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, position):

        real_index = self.indices[position]

        image, label = self.base_dataset[real_index]

        # PIL Image → NumPy
        image = np.array(image)

        # Apply MONAI transformations
        image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(
            int(label[0]),
            dtype=torch.long
        )

        return image, label


# ==========================================
# 5. CREATE 5 HOSPITAL DATASETS
# ==========================================

hospital_datasets = {}

for i in range(1, 6):

    hospital_indices = partitions[
        f"hospital_{i}"
    ]

    hospital_dataset = HospitalDataset(
        train_dataset,
        hospital_indices,
        medical_transform
    )

    hospital_datasets[
        f"hospital_{i}"
    ] = hospital_dataset

    print(
        f"Hospital {i} dataset created: "
        f"{len(hospital_dataset)} samples"
    )


# ==========================================
# 6. CREATE HOSPITAL DATALOADERS
# ==========================================

batch_size = 32

hospital_loaders = {}

for hospital_name, hospital_dataset in hospital_datasets.items():

    loader = DataLoader(
        hospital_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    hospital_loaders[
        hospital_name
    ] = loader


print("\nAll hospital DataLoaders created!")


# ==========================================
# 7. TEST ONE BATCH FROM EACH HOSPITAL
# ==========================================

print("\n================================")
print("MONAI HOSPITAL PIPELINE TEST")
print("================================")

for hospital_name, loader in hospital_loaders.items():

    images, labels = next(
        iter(loader)
    )

    print(f"\n{hospital_name}")

    print(
        "Image batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )

    print(
        "Image data type:",
        images.dtype
    )

    print(
        "Label data type:",
        labels.dtype
    )

# ==========================================
# 8. VISUALIZE TRANSFORMED MEDICAL IMAGE
# ==========================================

hospital_name = "hospital_1"

loader = hospital_loaders[hospital_name]

images, labels = next(iter(loader))

# Take first image from the batch
image = images[0]

# Remove channel dimension for visualization
image = image.squeeze(0)

plt.figure(figsize=(5, 5))

plt.imshow(
    image.numpy(),
    cmap="gray"
)

plt.title(
    f"{hospital_name} - "
    f"Label: {labels[0].item()}"
)

plt.axis("off")

plt.show()


# ==========================================
# 9. FINAL VERIFICATION
# ==========================================

print("\n================================")
print("FINAL PIPELINE VERIFICATION")
print("================================")

pipeline_success = True

for hospital_name, loader in hospital_loaders.items():

    images, labels = next(
        iter(loader)
    )

    if images.shape != (32, 1, 28, 28):

        pipeline_success = False

    if images.dtype != torch.float32:

        pipeline_success = False

    if labels.dtype != torch.long:

        pipeline_success = False


if pipeline_success:

    print(
        "Medical imaging pipeline: SUCCESS"
    )

    print(
        "All 5 hospitals produce "
        "model-ready PyTorch batches."
    )

else:

    print(
        "Medical imaging pipeline: FAILED"
    )