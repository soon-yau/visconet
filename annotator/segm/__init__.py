# Self-Correction-Human-Parsing
# Original https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

import os
import torch
import numpy as np
from PIL import Image
import cv2

import torchvision.transforms as T

from .transforms import transform_logits, get_affine_transform
from . import networks

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class Segmentator(torch.nn.Module):
    def __init__(self, dataset='lip'):
        super().__init__()

        if dataset == 'atr':
            model_path='models/exp-schp-201908301523-atr.pth'
        elif dataset == 'lip':
            model_path='models/exp-schp-201908261155-lip.pth'

        num_classes = dataset_settings[dataset]['num_classes']
        input_size = dataset_settings[dataset]['input_size']
        label = dataset_settings[dataset]['label']

        assert os.path.exists(model_path)
        self.model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
        state_dict = torch.load(model_path)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.palette = get_palette(num_classes)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale
    
    def preprocess(self, image:np.array):
        # convert numpy to cv2
        image = image[:,:,::-1]
        h, w, _ = image.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

    @torch.no_grad()
    def __call__(self, input_image):
        image, meta = self.preprocess(input_image)
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']
        input_size = list(self.input_size)
        device = next(self.parameters()).device
        output = self.model(image.unsqueeze(0).to(device))
        upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        #return output_img
        output_img.putpalette(self.palette)
        return output_img
        #return np.array(output_img)
        