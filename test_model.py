import torch #.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import FedMedModel, train, test


transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


trainloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

testloader = DataLoader(
    test_data,
    batch_size=32
)


model = FedMedModel()

print("Training started...")

train(
    model,
    trainloader,
    epochs=1
)

test_loss, accuracy = test(
    model,
    testloader
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")