import torch
import pickle
import numpy as np

def load_model(path):

    if path.endswith(".pth"):
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model, "torch"

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, "sklearn"


def predict(model, model_type, features):

    features = features.reshape(1, -1)

    if model_type == "torch":
        with torch.no_grad():
            pred = model(torch.tensor(features).float()).numpy()

    else:
        pred = model.predict(features)

    return pred
