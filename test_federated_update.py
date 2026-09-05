import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import FedMedModel, train_one_epoch
from model.parameter_io import get_model_parameters, set_model_parameters


# Hospital's local dataset
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


# Receive global model
global_model = FedMedModel()

global_parameters = get_model_parameters(global_model)

print("Global model parameters received.")


# Hospital creates its local model
local_model = FedMedModel()

set_model_parameters(
    local_model,
    global_parameters
)

print("Global parameters loaded into local model.")


# Local hospital training
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    local_model.parameters(),
    lr=0.001
)

loss = train_one_epoch(
    local_model,
    trainloader,
    optimizer,
    criterion
)

print(f"Local training loss: {loss:.4f}")


# Get updated parameters
updated_parameters = get_model_parameters(local_model)

print("Local model parameters updated.")

print("\n================================")
print("FEDERATED UPDATE TEST")
print("================================")
print("Global parameters:", len(global_parameters))
print("Updated parameters:", len(updated_parameters))
print("Parameter update generated successfully.")