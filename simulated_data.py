import os
import numpy as np

from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader


# =========================================
# 1. DEFINE NUMBER OF HOSPITALS
# =========================================

num_hospitals = 5


# ==========================================
# 2. CREATE HOSPITAL DIRECTORIES
# ==========================================

base_folder = "hospitals"

os.makedirs(base_folder, exist_ok=True)

for i in range(1, num_hospitals + 1):

    hospital_folder = os.path.join(
        base_folder,
        f"hospital_{i}"
    )

    os.makedirs(
        hospital_folder,
        exist_ok=True
    )


# ==========================================
# 3. LOAD MEDICAL TRAINING DATASET
# ==========================================

train_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = PneumoniaMNIST(
    split="train",
    transform=train_transform,
    download=True
)

print("Medical training dataset loaded!")
print("Total samples:", len(train_dataset))


# ==========================================
# 4. LOAD SAVED HOSPITAL PARTITIONS
# ==========================================

partitions = np.load(
    "hospital_partitions.npz"
)

print("\nHospital partitions loaded!")


# ==========================================
# 5. CREATE HOSPITAL DATASETS
# ==========================================

hospital_datasets = {}

for i in range(1, num_hospitals + 1):

    hospital_indices = partitions[
        f"hospital_{i}"
    ]

    hospital_dataset = Subset(
        train_dataset,
        hospital_indices
    )

    hospital_datasets[
        f"hospital_{i}"
    ] = hospital_dataset

    print(
        f"Hospital {i} dataset created: "
        f"{len(hospital_dataset)} samples"
    )


# ==========================================
# 6. VERIFY TOTAL DATA
# ==========================================

total_samples = 0

for hospital_name, dataset in hospital_datasets.items():

    total_samples += len(dataset)

print("\nTotal hospital samples:", total_samples)
print("Original training samples:", len(train_dataset))


if total_samples == len(train_dataset):

    print("Dataset verification: SUCCESS")

else:

    print("Dataset verification: FAILED")

# ==========================================
# 7. CREATE LOCAL DATALOADERS
# ==========================================

batch_size = 32

hospital_loaders = {}

for hospital_name, hospital_dataset in hospital_datasets.items():

    hospital_loader = DataLoader(
        hospital_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    hospital_loaders[hospital_name] = hospital_loader

    print(
        f"{hospital_name} DataLoader created"
    )
# ==========================================
# 8. TEST EACH HOSPITAL DATALOADER
# ==========================================

print("\nHospital Batch Verification")
print("---------------------------")

for hospital_name, hospital_loader in hospital_loaders.items():

    images, labels = next(iter(hospital_loader))

    print(f"\n{hospital_name}")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

# ==========================================
# 9. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nHospital Class Distribution")
print("---------------------------")

for hospital_name, hospital_dataset in hospital_datasets.items():

    normal_count = 0
    pneumonia_count = 0

    for index in hospital_dataset.indices:

        _, label = train_dataset[index]

        label_value = int(label[0])

        if label_value == 0:
            normal_count += 1

        elif label_value == 1:
            pneumonia_count += 1

    print(f"\n{hospital_name}")
    print("Normal:", normal_count)
    print("Pneumonia:", pneumonia_count)
    print(
        "Total:",
        normal_count + pneumonia_count
    )

# ==========================================
# 10. VERIFY HOSPITAL DATA ISOLATION
# ==========================================

print("\nHospital Isolation Check")
print("------------------------")

isolation_success = True

for i in range(1, num_hospitals + 1):

    current_indices = set(
        partitions[f"hospital_{i}"].tolist()
    )

    for j in range(i + 1, num_hospitals + 1):

        other_indices = set(
            partitions[f"hospital_{j}"].tolist()
        )

        overlap = current_indices.intersection(
            other_indices
        )

        if len(overlap) > 0:

            isolation_success = False

            print(
                f"Overlap found between "
                f"Hospital {i} and Hospital {j}"
            )

if isolation_success:

    print("Hospital isolation check: SUCCESS")
    print("No samples are shared between hospitals.")

else:

    print("Hospital isolation check: FAILED")

# ==========================================
# 11. FINAL DELIVERABLE 3 VERIFICATION
# ==========================================

print("\n========================================")
print("DELIVERABLE 3 FINAL VERIFICATION")
print("========================================")

print("Number of hospitals:", len(hospital_datasets))

for hospital_name, hospital_dataset in hospital_datasets.items():

    print(
        f"{hospital_name}: "
        f"{len(hospital_dataset)} samples"
    )

print("\nTotal samples:", sum(
    len(dataset)
    for dataset in hospital_datasets.values()
))

print("Original training samples:", len(train_dataset))

print("\nLocal DataLoaders:", len(hospital_loaders))

if (
    len(hospital_datasets) == 5
    and
    len(hospital_loaders) == 5
    and
    sum(len(dataset) for dataset in hospital_datasets.values())
    == len(train_dataset)
):

    print("\nDELIVERABLE 3: COMPLETED SUCCESSFULLY")

else:

    print("\nDELIVERABLE 3: VERIFICATION FAILED")