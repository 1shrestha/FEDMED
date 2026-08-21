import medmnist
from medmnist import PneumoniaMNIST

train_dataset = PneumoniaMNIST(
    split="train",
    download=True
)

val_dataset = PneumoniaMNIST(
    split="val",
    download=True
)

test_dataset = PneumoniaMNIST(
    split="test",
    download=True
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))