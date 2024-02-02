import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat

class LinearProj(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            context_dim=None,
        ):
        super().__init__()

        self.proj = nn.Linear(768, context_dim)
            
    def forward(self, x):
        return  self.proj(x)

class ProjectLocalStyle(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            pool_size=8,
            local_emb_size=257,
            bias=False            
        ):
        super().__init__()
        self.pool_size = pool_size
        self.local_emb_size = local_emb_size

        self.proj = nn.Linear(self.local_emb_size, self.pool_size, bias=bias)
            
    def forward(self, x):
        b, n, c, d = x.shape
        # reaarange [2, 8, 257, 1024] to [2, 8, 1024, 257]
        batch = rearrange(x, 'b n c d -> b n d c')
        # linear (257, 8) to create [2, 8, 1024, 8]
        batch = self.proj(batch)
        # reaarange [2, 8, 1024, 8] to [2, 8x8, 1024]
        b, n, c, p = batch.shape
        batch = rearrange(batch, 'b n c p -> b (n p) c')
        return batch



class ClipImageEncoder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model='ViT-L/14',
            context_dim=None,
            jit=False,
            device='cuda',
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(768, context_dim)
            
    @torch.no_grad()
    def extract_features(self, x):
        b, n, c, h, w = x.shape
        batch = rearrange(x, 'b n c h w -> (b n) c h w ')
        ret = self.model.encode_image(batch)
        return ret
            
    def forward(self, x):
        b, n, c, h, w = x.shape
        ret = self.extract_features(x)
        ret = self.proj(ret.float())
        return rearrange(ret, '(b n) w -> b n w ', b=b, n=n)
