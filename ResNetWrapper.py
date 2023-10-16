from typing import Type, Union, Optional, List, Any
import wandb
import torch


from ResNet18 import resnet18


def log_intermediate_representations(batch):
    wandb.login()
    wandb.init(project="ResNet18 Demo", mode="online")

    model = resnet18()
    model.eval()

    pre_iht_layer_inputs = []

    def get_iht_layer_data(module, output):
        pre_iht_layer_inputs.append(output[0])

    pre_iht_layer = model._modules.get('htiht')._modules.get('iht')
    _ = pre_iht_layer.register_forward_pre_hook(get_iht_layer_data)

    out = model(batch)

    wandb.finish()
