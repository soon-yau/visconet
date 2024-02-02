#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat

from glob import glob
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import pickle
import os
import numpy as np

# In[17]:


device = 'cuda:0'

clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
clip_transform = T.Compose([T.ToTensor(),
                            clip_norm])


# In[1]:


class ClipImageEncoder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model='ViT-L/14',
            context_dim=[],
            jit=False,
            device='cuda',
        ):
        super().__init__()
        self.context_dim = context_dim
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    @torch.no_grad()        
    def forward(self, x):
        b, n, c, h, w = x.shape
        batch = rearrange(x, 'b n c h w -> (b n) c h w ')
        ret = self.model.encode_image(batch)        
        return rearrange(ret, '(b n) w -> b n w ', b=b, n=n)

    def preprocess(self, style_file):
        if os.path.exists(style_file):
            style_image = Image.open(style_file)
        else:
            style_image = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
        x = clip_transform(style_image).unsqueeze(0).unsqueeze(0)
        return x

    def postprocess(self, x):
        return x.squeeze(0).detach().cpu().numpy()
# In[23]:


encoder = ClipImageEncoder()
encoder = encoder.to(device)


# In[6]:


# style_files = glob("/home/soon/datasets/deepfashion_inshop/styles/**/*.jpg", recursive=True)


# # In[39]:


# for style_file in tqdm(style_files[:]):
#     style_image = Image.open(style_file)
#     x = clip_transform(style_image).unsqueeze(0).unsqueeze(0).to(device)    
#     emb = encoder(x).detach().cpu().squeeze(0).numpy()
#     emb_file = style_file.replace('.jpg','.p')
#     with open(emb_file, 'wb') as file:
#         pickle.dump(emb, file)    
