import inspect
import types

import torch
import torchvision
from torchvision.models import list_models, get_model
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from . import resnet
from . import convnext


# Lister les fonctions définies dans le module
MODELS = {
    name: func
    for name, func in inspect.getmembers(resnet)
    if isinstance(func, types.FunctionType) and func.__module__ == resnet.__name__
}

MODELS.update(
    {
        name: func
        for name, func in inspect.getmembers(convnext)
        if isinstance(func, types.FunctionType) and func.__module__ == convnext.__name__
    }
)

MODELS.update({name: cls for name, cls in inspect.getmembers(resnet, inspect.isclass)})
MODELS.update({name: cls for name, cls in inspect.getmembers(convnext, inspect.isclass)})


def build(name, return_nodes, load=False, source="torchvision", **cfg):
    """
    Build a backbone model based on the provided configuration.

    Args:
        name: str: Name of the backbone model to build (e.g., 'resnext50', 'convnextv2_nano').
        return_nodes: name of the layers returned by the model
        load either False of the path where the state_dict is saved
        source: either 'torchvision' or 'local' for locally defined models
        cfg (dict): Configuration dictionary containing model parameters.

    Returns:
        nn.Module: The constructed backbone model.
    """
    if source == "torchvision":
        classification_models = list_models(module=torchvision.models)
        if name not in classification_models:
            raise ValueError(f"name '{name}' not found. Available local models: {classification_models}")
        model = get_model(name, weights="DEFAULT")

    else:
        if name not in MODELS:
            raise ValueError(f"name '{name}' not found. Available local models: {list(MODELS.keys())}")
        print(f"Building model with config: {cfg}")
        model = MODELS[name](**cfg)

    if load:
        try:
            model.load_state_dict(load)
        except:
            model.load_state_dict(torch.load(load)["model"])

    return create_feature_extractor(model, return_nodes)
