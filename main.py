import gc
import numpy as np
import cv2
import random
import torch.nn as nn

import torch

from ResNet18 import resnet18
from preprocess_input import ImageTransformation
from imagenet_classes import ImageNetClasses
from gradient_based_saliency_map import visualize_grads
from config import CONFIG


def preprocess_input(img_path: str):
    img = cv2.imread(img_path)
    transform_pipeline = ImageTransformation()
    return transform_pipeline(img)


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


def predict(img_path: str, device: str, pretrained: bool = True, requires_grad: bool = True):
    preprocessed_input, batch = preprocess_input(img_path)
    model = resnet18(pretrained=pretrained).to(device)
    model.eval()

    batch = batch[None, :, :, :].to(device)
    if requires_grad:
        batch.requires_grad = True
        model_out = model(batch)
    else:
        with torch.no_grad():
            model_out = model(batch) 

    predicted_class_idx = torch.argmax(model_out)
    probabilities = nn.Softmax(dim=1)(model_out)[0]
    print(f'Predicted class: {ImageNetClasses[predicted_class_idx.item()]} with probability {torch.max(probabilities)}')

    if requires_grad:
        one_hot_encoding = nn.functional.one_hot(torch.tensor([predicted_class_idx]), num_classes=1000).to(device)
        grads = get_gradient_map(model_out, batch, one_hot_encoding)
        grads = grads.detach().cpu().numpy()
        saliency_map = visualize_grads(preprocessed_input, grads)
        cv2.imshow("Gradient based saliency map", cv2.hconcat([saliency_map, np.array(preprocessed_input)]))
        cv2.waitKey()

def export_to_onnx(img_path: str):
    _, batch = preprocess_input(img_path)
    batch = batch[None, :, :, :]
    model = resnet18(pretrained=True)
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
        accelerator = 'cuda'

    print(f"Accelerator: {accelerator}")

    # img_path = './5KQYCLPKNCCZ.jpg'
    img_path = './dog.jpg'
    predict(img_path, accelerator)


if __name__ == "__main__":
    main()