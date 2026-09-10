import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model.model import FedMedModel, train_one_epoch, test
from model.parameter_io import get_model_parameters, set_model_parameters


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


# Simulated hospital datasets
hospital1 = Subset(train_data, range(0, 1000))
hospital2 = Subset(train_data, range(1000, 2000))

loader1 = DataLoader(hospital1, batch_size=32, shuffle=True)
loader2 = DataLoader(hospital2, batch_size=32, shuffle=True)


# Global model
global_model = FedMedModel()

global_parameters = get_model_parameters(global_model)

criterion = torch.nn.CrossEntropyLoss()


# Hospital 1 local training
model1 = FedMedModel()
set_model_parameters(model1, global_parameters)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

loss1 = train_one_epoch(
    model1,
    loader1,
    optimizer1,
    criterion
)

params1 = get_model_parameters(model1)


# Hospital 2 local training
model2 = FedMedModel()
set_model_parameters(model2, global_parameters)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

loss2 = train_one_epoch(
    model2,
    loader2,
    optimizer2,
    criterion
)

params2 = get_model_parameters(model2)


# Server aggregation
aggregated_parameters = []

for p1, p2 in zip(params1, params2):
    aggregated_parameters.append((p1 + p2) / 2)

set_model_parameters(
    global_model,
    aggregated_parameters
)


# Evaluate new global model
testloader = DataLoader(
    test_data,
    batch_size=32
)

test_loss, accuracy = test(
    global_model,
    testloader
)


print("\n================================")
print("DAY 10 END-TO-END FEDERATED TEST")
print("================================")
print(f"Hospital 1 loss: {loss1:.4f}")
print(f"Hospital 2 loss: {loss2:.4f}")
print(f"Global Test Loss: {test_loss:.4f}")
print(f"Global Test Accuracy: {accuracy:.2f}%")
print("Federated round completed successfully.")