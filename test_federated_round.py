import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model.model import FedMedModel, train_one_epoch
from model.parameter_io import get_model_parameters, set_model_parameters


# Dataset
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Two simulated hospitals
hospital1 = Subset(dataset, range(0, 1000))
hospital2 = Subset(dataset, range(1000, 2000))

loader1 = DataLoader(hospital1, batch_size=32, shuffle=True)
loader2 = DataLoader(hospital2, batch_size=32, shuffle=True)


# Global model
global_model = FedMedModel()

global_parameters = get_model_parameters(global_model)

print("Global model created.")


# Hospital 1
model1 = FedMedModel()
set_model_parameters(model1, global_parameters)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

loss1 = train_one_epoch(
    model1, loader1, optimizer1, criterion
)

params1 = get_model_parameters(model1)

print(f"Hospital 1 loss: {loss1:.4f}")


# Hospital 2
model2 = FedMedModel()
set_model_parameters(model2, global_parameters)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

loss2 = train_one_epoch(
    model2, loader2, optimizer2, criterion
)

params2 = get_model_parameters(model2)

print(f"Hospital 2 loss: {loss2:.4f}")


# Federated averaging
averaged_parameters = []

for p1, p2 in zip(params1, params2):
    averaged_parameters.append((p1 + p2) / 2)


# Update global model
set_model_parameters(
    global_model,
    averaged_parameters
)

print("Global model updated using both hospitals.")

print("\n==============================")
print("DAY 7 FEDERATED ROUND TEST")
print("==============================")
print("Hospitals trained locally: 2")
print("Global model updated: SUCCESS")
print("Federated averaging: SUCCESS")