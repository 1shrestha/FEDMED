import torch
import torch.nn as nn
import torch.optim as optim


class FedMedModel(nn.Module):

    def __init__(self):
        super(FedMedModel, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


def train(model, trainloader, epochs=1):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for images, labels in trainloader:

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(trainloader):.4f}"
        )


def test(model, testloader):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in testloader:

            output = model(images)

            loss = criterion(output, labels)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(testloader)

    return average_loss, accuracy


def train_one_epoch(
    model,
    trainloader,
    optimizer,
    criterion,
    device="cpu"
):

    model.train()

    total_loss = 0.0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(trainloader)