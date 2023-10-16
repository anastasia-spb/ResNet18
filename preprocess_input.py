import numpy as np
import torch
from torchvision import transforms


def bgr2rgb(image):
    return image[:,:,::-1]


def normalize(frame, means, stds):
    frame = (frame.astype('float32') - means) / stds
    return frame

class ImageTransformation(object):
    # [0.4850, 0.4560, 0.4060] in RGB format
    means = [123.675, 116.280, 103.530]
    # [0.2290, 0.2240, 0.2250] in RGB format
    stds = [58.395, 57.120, 57.375]
    input_size = 235
    crop_size = (224, 224)

    def __call__(self, frame: np.array, to_rgb: bool = True):
        if to_rgb:
            frame = bgr2rgb(frame)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=self.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.crop_size)])

        frame = transform(frame)
        normalized_frame = normalize(np.array(frame), self.means, self.stds)
        normalized_frame = normalized_frame.transpose(2, 0, 1)
        normalized_frame = torch.as_tensor(np.ascontiguousarray(normalized_frame))

        return frame, normalized_frame.type(torch.FloatTensor)