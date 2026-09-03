import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import FedMedModel, train_one_epoch


transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

model = FedMedModel()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Day 3: Local Training")
print("----------------------")

loss = train_one_epoch(
    model,
    trainloader,
    optimizer,
    criterion
)

print(f"Local training loss: {loss:.4f}")
print("Local training completed successfully.")