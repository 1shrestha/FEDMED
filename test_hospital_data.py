from data.hospital_data import get_hospital_loader


print("DAY 4: Hospital DataLoader Test")
print("--------------------------------")

for hospital_id in range(1, 6):

    loader = get_hospital_loader(
        hospital_id,
        batch_size=32
    )

    print(
        f"Hospital {hospital_id}: "
        f"{len(loader.dataset)} samples, "
        f"{len(loader)} batches"
    )

print("\nHospital-specific DataLoaders working successfully.")
