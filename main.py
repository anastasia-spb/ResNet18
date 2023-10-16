import os
from astropy.convolution import convolve, Gaussian2DKernel
import gc
import numpy as np
import cv2
import random
import torch.nn as nn

from typing import Union

import torch

from ResNet18 import resnet18
from preprocess_input import ImageTransformation
from config import CONFIG


def preprocess_input(img_path: str, device: str):
    img = cv2.imread(img_path)
    transform_pipeline = ImageTransformation()
    return transform_pipeline(img, device)

def get_gradient_map(outputs: torch.Tensor, inputs: torch.Tensor, one_hot_target: torch.Tensor):
    """
    Wraps https://pytorch.org/docs/stable/generated/torch.autograd.grad.html function
    Args:
        outputs: Model outputs
        inputs: Inputs w.r.t. which the gradient will be returned
        one_hot_target: Gradients with respect to heads
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=one_hot_target,
        retain_graph=True)[0]


def get_heatmap(gradients: np.ndarray):
    heatmap = np.mean(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.amax(heatmap)
    smoothed_heatmap = convolve(heatmap, Gaussian2DKernel(x_stddev=5))
    smoothed_heatmap /= np.linalg.norm(smoothed_heatmap)
    return smoothed_heatmap

def visualize_segmentation_probabilities(image, probabilities, color_map: Union[int, str, np.array] = cv2.COLORMAP_JET,
                                         alpha=0.8, inplace=True) -> np.ndarray:
    assert len(probabilities.shape) == 2
    assert image.shape[0] == probabilities.shape[0]
    assert image.shape[1] == probabilities.shape[1]

    segmentation = cv2.applyColorMap((probabilities * 255).astype(np.uint8), color_map)
    probabilities = probabilities[:, :, np.newaxis] * alpha

    out = (image * (1 - probabilities) + probabilities * segmentation).astype(np.uint8)
    if inplace:
        image[:] = out[:]
    return out


def visualize_grads(preprocessed_input, grads: np.array):
    heatmap = get_heatmap(grads)
    heatmap = cv2.resize(heatmap, (preprocessed_input.shape[1], preprocessed_input.shape[0]),
                         interpolation=cv2.INTER_AREA)
    probabilities = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
    return visualize_segmentation_probabilities(preprocessed_input, probabilities, alpha=1.0, inplace=True)


def predict(img_path: str, device: str, pretrained: bool = True, requires_grad: bool = True):
    preprocessed_input, batch = preprocess_input(img_path, device)
    model = resnet18(pretrained=pretrained).to(device)
    model.eval()
    if requires_grad:
        batch.requires_grad = True
        model_out = model(batch)
    else:
        with torch.no_grad():
            model_out = model(batch)

    if requires_grad:
        one_hot_encoding = nn.functional.one_hot(torch.tensor(inputs), num_classes=1000).to(device)
        grads = get_gradient_map(model_out, batch, one_hot_encoding)
        grads = grads.detach().cpu().numpy()
        saliency_map = visualize_grads(preprocessed_input, grads)
        cv2.imshow("Gradient based saliency map", cv2.hconcat([saliency_map, preprocessed_input]))
        cv2.waitKey()

def export_to_onnx(img_path: str):
    _, batch = preprocess_input(img_path, device='cpu')
    model = resnet18(pretrained=True).to('cpu')
    model.eval()

    torch.onnx.export(model, batch, './resnet18.onnx')


def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    accelerator = 'cpu'
    if torch.cuda.is_available():
        # Cuda maintenance
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(CONFIG['seed'])
        accelerator = 'gpu'

    print(f"Accelerator: {accelerator}")

    img_path = './broccoli.jpeg'
    predict(img_path, accelerator)


if __name__ == "__main__":
    main()