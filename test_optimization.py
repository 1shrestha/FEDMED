import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import FedMedModel, train_one_epoch


transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

criterion = torch.nn.CrossEntropyLoss()

model = FedMedModel()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss = train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
)

print("DAY 9 MODEL OPTIMIZATION")
print("------------------------")
print(f"Learning rate: {0.001}")
print(f"Training loss: {loss:.4f}")
print("Optimized local training completed successfully.")