import os
import numpy as np
from deepface import DeepFace

from PIL import Image
from collections import OrderedDict
from torchvision import transforms as T
import torch
from annotator.segm import Segmentator

class SegmentCropper:
    def __init__(self, 
                 label_dict: dict, 
                 segm_groups: dict, 
                 image_transform,
                 feat_transform=None,
                 device=None):
        self.image_transform = image_transform
        self.feat_transform = feat_transform
        self.device = device
        self.label_dict = label_dict
        self.label2id = dict(zip(self.label_dict.values(), self.label_dict.keys()))
        self.segm_groups = segm_groups
        self.segm_id_groups = OrderedDict()
        for k, v in self.segm_groups.items():
            self.segm_id_groups[k] = [self.label2id[l] for l in v]
                
    def get_mask(self, segm, mask_val, default_value=1.0):
        mask = np.full(segm.shape, default_value, dtype=np.float32)
        if mask_val:
            for label, value in mask_val.items():
                mask[segm==self.label2id[label]] = value
        return mask       
    
    
    def get_binary_mask(self, segm, mask_ids):
        mask = np.full(segm.shape, False)
        for mask_id in mask_ids:
            mask |= segm == mask_id    
        return mask        
        
    def get_mask_range(self, mask, margin):
        height, width = mask.shape
        left = 0
        right = width
        top = 0
        bottom = height
        
        vertical = torch.sum(mask.to(torch.float32), dim=0)
        for w in range(width):
            if vertical[w] > 0.1:
                left = w
                break

        for w in range(width-1, -1, -1):
            if vertical[w] > 0.1:
                right = w
                break

        left = max(0, left-margin)
        right = min(width, right+margin)

        horizontal = torch.sum(mask.to(torch.float32), dim=1)

        for h in range(height):
            if horizontal[h] > 0.1:
                top = h
                break

        for h in range(height-1, -1, -1):
            if horizontal[h] > 0.1:
                bottom = h
                break

        top = max(0, top-margin)
        bottom = min(height, bottom+margin)

        if left==right or top==bottom:
            return None
        return {'left':left, 'right':right, 'top':top, 'bottom':bottom}

    def crop(self, 
             input_image: torch.tensor,
             mask: np.array, 
             margin=0,
             is_background=False,
             mask_background=False,
             name=None):
        
        image = torch.clone(input_image).detach()
        mask_range = self.get_mask_range(mask, margin)
        if mask_range == None:
            return None
        if is_background: # fill the mask with average background color
            new_images = []
            for i in range(3):
                mean_color = torch.masked_select(image[i], mask==True).mean()
                new_images.append(image[i].masked_fill(mask==False, mean_color))

            cropped = torch.stack(new_images)
        else:    
            if name=='face':
                if (mask_range['bottom']-mask_range['top'])<32 or \
                (mask_range['right']-mask_range['left'])<32:
                #if ((mask_range['bottom']-mask_range['top'])<128) ||
                #   ((mask_range['right']-mask_range['left'])<128):
                    return None #torch.ones((3, 224,224))

            cropped = image * mask if mask_background else image
            detected =  cropped.sum() > 0
            grey_shade = 1.0
            if mask_background:                
                grey_shade = 0.7
                grey_bg = (grey_shade * torch.ones_like(image)) * ~mask
                cropped += grey_bg                

            cropped = cropped[:,mask_range['top']:mask_range['bottom'], 
                                mask_range['left']:mask_range['right']]

            if detected:
                _, h, w = cropped.shape
                pad = (h - w)//2
                if pad > 0:
                    padding  = (pad, pad, 0, 0)
                    cropped = torch.nn.functional.pad(cropped, padding,value=grey_shade)
                    #cropped = torch.nn.ZeroPad2d(padding)(cropped)
                elif pad < 0:
                    padding  = (0, 0, -pad, -pad)
                    cropped = torch.nn.functional.pad(cropped, padding,value=grey_shade)
                    #cropped = torch.nn.ZeroPad2d(padding)(cropped)                
            else:
                return None #return torch.ones((3, 224,224))

        cropped = self.image_transform(cropped)
        if self.feat_transform:
            cropped = self.feat_transform(cropped)
        return cropped
    
    def __call__(self, image:np.array, segm:np.array=None, ignore_head=False, ignore_hair=False):

        if segm == None:
            segm = np.array(self.segmentator(image))
        
        cropped_images =  OrderedDict()
        face_mask = torch.zeros(image.shape[:2], dtype=torch.bool).to(self.device)
        try:
            face = DeepFace.extract_faces(img_path=image[:,:,::-1], 
                                    target_size=(224, 224), 
                                    detector_backend='dlib',
                                    enforce_detection=True,
                                    align=True)[0]
            if face['confidence'] > 0.1:
                cropped_images['face'] = \
                    Image.fromarray(np.uint8(face['face']*255))

                face_coords = face['facial_area']
                face_mask[face_coords['y']:face_coords['y']+face_coords['h'],
                          face_coords['x']:face_coords['x']+face_coords['w']] = True
                
        except:
            pass


        image = T.ToTensor()(image)
        if self.device:
            image = image.to(self.device)
        
        for name, segm_group in self.segm_id_groups.items():
            mask = torch.from_numpy(self.get_binary_mask(segm, segm_group)).to(self.device)
            if name == 'background':
                human_mask = (~mask) * 1
                #cropped_images['human_mask'] = (~mask) * 1
            elif name == 'face':
                if ignore_head:
                    human_mask ^= (mask & face_mask)
            else:
                if name =='hair' and ignore_hair:
                    human_mask ^= mask
                cropped = self.crop(image, mask, 
                                    margin=0,#margins[name],
                                    is_background=name=='background',
                                    name=name,
                                    mask_background=True, #name!='face'
                                    )
                cropped_images[name] = cropped
        
        cropped_images['human_mask'] = human_mask
        return cropped_images
    
    def to(self, device):
        self.segmentator = self.segmentator.to(device)

class ATRSegmentCropper(SegmentCropper):
    def __init__(self, feat_transform=None, **kwargs):
        label_dict = {
             0: 'Background',
             1: 'Hat',
             2: 'Hair',
             3: 'Sunglasses',
             4: 'Upper-clothes',
             5: 'Skirt',
             6: 'Pants',
             7: 'Dress',
             8: 'Belt',
             9: 'Left-shoe',
             10: 'Right-shoe',
             11: 'Face',
             12: 'Left-leg',
             13: 'Right-leg',
             14: 'Left-arm',
             15: 'Right-arm',
             16: 'Bag',
             17: 'Scarf'}
        segm_groups = {
            'background':['Background'],
            'face':['Sunglasses','Face'],
            'hair':['Hair'],
            'headwear':['Hat'],
            'top':['Upper-clothes','Dress'],
            #'outer':['Upper-clothes'],
            'bottom':['Skirt','Pants','Dress','Belt'],
            'shoes':['Left-shoe','Right-shoe'],
            #'accesories':['Bag'],
            }        
            
        image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224))])
        
        self.segmentator = Segmentator('atr')

        super().__init__(label_dict, segm_groups, image_transform, feat_transform, **kwargs)

class LipSegmentCropper(SegmentCropper):
    def __init__(self, feat_transform=None, **kwargs):
        label_dict = {
             0: 'Background',
             1: 'Hat',
             2: 'Hair',
             3: 'Glove',
             4: 'Sunglasses',
             5: 'Upper-clothes',
             6: 'Dress',
             7: 'Coat',
             8: 'Socks',
             9: 'Pants', 
             10: 'Jumpsuits',
             11: 'Scarf',
             12: 'Skirt', 
             13: 'Face',
             14: 'Left-arm',
             15: 'Right-arm',
             16: 'Left-leg',
             17: 'Right-leg',
             18: 'Left-shoe', 
             19: 'Right-shoe'}
        segm_groups = {
            'background':['Background'],
             #'face':['Sunglasses','Face'],
            'hair':['Hair'],
            'headwear':['Hat'],
            'top':['Upper-clothes','Dress','Jumpsuits'],
            'outer':['Coat'],
            'bottom':['Skirt','Pants','Dress','Jumpsuits'],
            'shoes':['Left-shoe','Right-shoe'],
            #'accesories':['Bag'],
            }        
            
        image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224))])
        
        self.segmentator = Segmentator('lip')

        super().__init__(label_dict, segm_groups, image_transform, feat_transform, **kwargs)
