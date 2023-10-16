from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np
import cv2


from typing import Union


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
    heatmap = get_heatmap(grads[0])
    heatmap = cv2.resize(heatmap, (preprocessed_input.size[1], preprocessed_input.size[0]),
                         interpolation=cv2.INTER_AREA)
    probabilities = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
    return visualize_segmentation_probabilities(np.array(preprocessed_input), probabilities, alpha=1.0, inplace=True)
