import torch

from model.model import FedMedModel
from model.parameter_io import (
    get_model_parameters,
    set_model_parameters,
    save_model,
    load_model,
)


print("Creating model...")

model = FedMedModel()

# Get model parameters
parameters = get_model_parameters(model)

print("\nParameter handling test")
print("-----------------------")

print("Number of parameter tensors:", len(parameters))

for i, parameter in enumerate(parameters):
    print(
        f"Parameter {i}: "
        f"shape={tuple(parameter.shape)}, "
        f"dtype={parameter.dtype}"
    )


# Create a second model
model2 = FedMedModel()

# Load parameters into second model
set_model_parameters(model2, parameters)

print("\nParameters successfully loaded into second model.")


# Verify both models have identical parameters
same = True

for p1, p2 in zip(model.parameters(), model2.parameters()):
    if not torch.equal(p1, p2):
        same = False
        break


print("Models identical:", same)


# Test saving
save_model(model, "fedmed_model.pth")

print("Model saved successfully.")


# Test loading
model3 = FedMedModel()
load_model(model3, "fedmed_model.pth")

print("Model loaded successfully.")


print("\n================================")
print("DAY 2 PARAMETER I/O TEST")
print("================================")

if same:
    print("SUCCESS: Model parameter handling works correctly.")
else:
    print("FAILED: Parameter mismatch.")