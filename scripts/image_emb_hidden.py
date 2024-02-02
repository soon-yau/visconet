#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import torch.nn as nn
from functools import partial
#import clip
from einops import rearrange, repeat

from glob import glob
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import pickle
import numpy as np
import os

from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPProcessor, CLIPModel
device = 'cuda:0'

#model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

class ClipImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = (1, 257, 1024)
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()        
    def forward(self, x):
        ret = self.model(x)
        return ret.last_hidden_state, ret.image_embeds

    def preprocess(self, style_image):
        # if os.path.exists(style_file):
        #     style_image = Image.open(style_file)
        # else:
        #     style_image = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
        x = torch.tensor(np.array(self.processor.image_processor(style_image).pixel_values))
        return x

    def postprocess(self, x): # return numpy
        return x.detach().cpu().squeeze(0).numpy()

if __name__ == '__main__':
    device = 'cuda:1'
    style_files = glob("/home/soon/datasets/deepfashion_inshop/styles_default/**/*.jpg", recursive=True)
    style_files = [x for x in style_files if x.split('/')[-1]!='background.jpg']
    clip_model = ClipImageEncoder().to(device)

    for style_file in tqdm(style_files[24525:]):
        style_image = Image.open(style_file)
        emb_local, emb_global = clip_model(clip_model.preprocess(style_image).to(device))
        emb_local = clip_model.postprocess(emb_local)
        emb_global = clip_model.postprocess(emb_global)
        #x = torch.tensor(np.array(processor.image_processor(style_image).pixel_values))
        #emb = model(x.to(device)).last_hidden_state
        #emb = emb.detach().cpu().squeeze(0).numpy()
        emb_file = style_file.replace('.jpg','_hidden.p')
        with open(emb_file, 'wb') as file:
            pickle.dump(emb_local, file)    
        emb_file = style_file.replace('.jpg','.p')
        with open(emb_file, 'wb') as file:
            pickle.dump(emb_global, file)    
