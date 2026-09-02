import torch


def get_model_parameters(model):
    """
    Extract model parameters as a list of tensors.
    Used when sending model parameters to the federated server.
    """
    return [
        param.detach().cpu().clone()
        for param in model.state_dict().values()
    ]


def set_model_parameters(model, parameters):
    """
    Load a list of tensors into the model.
    Used when receiving global model parameters.
    """
    state_dict = model.state_dict()

    new_state_dict = {
        key: value.clone()
        for key, value in zip(state_dict.keys(), parameters)
    }

    model.load_state_dict(new_state_dict)


def save_model(model, path):
    """Save model parameters to disk."""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Load model parameters from disk."""
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)

    return model