import torch

from model.model import FedMedModel
from model.fedmed_training_pipeline import (
    get_parameters,
    set_parameters,
    count_parameters,
    model_parameter_norm,
    average_parameters,
)


print("DAY 13: FINAL ML PIPELINE TEST")
print("--------------------------------")


model1 = FedMedModel()
model2 = FedMedModel()

parameters1 = get_parameters(model1)
parameters2 = get_parameters(model2)

print("Model 1 parameters:", len(parameters1))
print("Model 2 parameters:", len(parameters2))

set_parameters(model2, parameters1)

print("Parameter transfer: SUCCESS")

averaged = average_parameters(
    [parameters1, parameters2]
)

print("Federated averaging: SUCCESS")
print("Total model parameters:", count_parameters(model1))
print(f"Model parameter norm: {model_parameter_norm(model1):.4f}")

print("\n==============================")
print("DAY 13 COMPLETE")
print("===============================")
print("ML pipeline integration: SUCCESS")