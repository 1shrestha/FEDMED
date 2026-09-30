from medmnist import PneumoniaMNIST
from torch.utils.data import Subset
import numpy as np


# ==========================================
# 1. LOAD TRAINING DATASET
# ==========================================

train_dataset = PneumoniaMNIST(
    split="train",
    download=True
)

print("Training dataset loaded!")
print("Total training samples:", len(train_dataset))


# ==========================================
# 2. DEFINE NUMBER OF HOSPITALS
# ==========================================

num_hospitals = 5

print("Number of hospitals:", num_hospitals)


# ==========================================
# 3. CREATE SAMPLE INDICES
# ==========================================

indices = np.arange(len(train_dataset))

print("Total indices:", len(indices))


# ==========================================
# 4. SHUFFLE INDICES
# ==========================================

np.random.seed(42)

np.random.shuffle(indices)

print("Dataset indices shuffled successfully!")


# ==========================================
# 5. SPLIT INDICES INTO HOSPITALS
# ==========================================

hospital_indices = np.array_split(
    indices,
    num_hospitals
)


# ==========================================
# 6. DISPLAY HOSPITAL SAMPLE COUNTS
# ==========================================

print("\nHospital Data Distribution")
print("--------------------------")

for i, indices_for_hospital in enumerate(hospital_indices):

    print(
        f"Hospital {i + 1}: "
        f"{len(indices_for_hospital)} samples"
    )

# ==========================================
# 7. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nClass Distribution Per Hospital")
print("--------------------------------")

for i, indices_for_hospital in enumerate(hospital_indices):

    labels = []

    for index in indices_for_hospital:
        _, label = train_dataset[index]
        labels.append(int(label[0]))

    normal_count = labels.count(0)
    pneumonia_count = labels.count(1)

    print(f"\nHospital {i + 1}")
    print("Normal:", normal_count)
    print("Pneumonia:", pneumonia_count)
    print("Total:", len(labels))

# ==========================================
# 8. CREATE HOSPITAL DATASETS
# ==========================================

hospital_datasets = []

for i, indices_for_hospital in enumerate(hospital_indices):

    hospital_dataset = Subset(
        train_dataset,
        indices_for_hospital
    )

    hospital_datasets.append(hospital_dataset)

    print(
        f"Hospital {i + 1} dataset created: "
        f"{len(hospital_dataset)} samples"
    )

# ==========================================
# 9. VERIFY HOSPITAL DATASETS
# ==========================================

print("\nFinal Hospital Dataset Verification")
print("------------------------------------")

total_samples = 0

for i, hospital_dataset in enumerate(hospital_datasets):

    sample_count = len(hospital_dataset)

    total_samples += sample_count

    print(
        f"Hospital {i + 1}: "
        f"{sample_count} samples"
    )

print("\nTotal samples across hospitals:", total_samples)
print("Original training samples:", len(train_dataset))

if total_samples == len(train_dataset):
    print("Partition verification: SUCCESS")
else:
    print("Partition verification: FAILED")

# ==========================================
# 10. SAVE HOSPITAL PARTITIONS
# ==========================================

np.savez(
    "hospital_partitions.npz",
    hospital_1=hospital_indices[0],
    hospital_2=hospital_indices[1],
    hospital_3=hospital_indices[2],
    hospital_4=hospital_indices[3],
    hospital_5=hospital_indices[4]
)

print("\nHospital partitions saved successfully!")

# ==========================================
# 11. FINAL DELIVERABLE 2 CHECK
# ==========================================

print("\n================================")
print("DELIVERABLE 2 COMPLETED")
print("Dataset partitioning successful")
print("5 hospital partitions created")
print("All samples verified")
print("Partitions saved")
print("================================")