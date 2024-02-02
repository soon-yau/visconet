from cldm.cldm import ControlNet, ControlLDM
import os
import einops
import torch
import torch as th
import torch.nn as nn
from torchvision import transforms as T
from pathlib import Path

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ViscoNetLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, control_cond_config, control_crossattn_key, mask_key=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.control_crossattn_key = control_crossattn_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.mask_enables = [1] * 13
        self.mask_key = mask_key

        # new
        self.control_cond_model = instantiate_from_config(control_cond_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c_text = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        ret_dict = dict(c_text=[c_text], c_concat=[control])

        def format_input(key):
            val = batch[key]
            if bs is not None:
                val = val[:bs]
            val = val.to(memory_format=torch.contiguous_format).float()
            val = val.to(self.device)
            return val
 
        if self.control_crossattn_key:
            ret_dict['c_crossattn']=[format_input(self.control_crossattn_key)]

        if self.mask_key:
            ret_dict['c_concat_mask']=[format_input(self.mask_key)]

        return x, ret_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # c_concat : skeleton [batch, 3, 512, 512]
        # c_crossattn : text [batch, 77, 1024]
        cond_txt = torch.cat(cond['c_text'], 1) # remove list
        cond_cross = torch.cat(cond['c_crossattn'], 1) # remove list
        cond_mask = torch.cat(cond['c_concat_mask'], 1)
        cond_concat = torch.cat(cond['c_concat'], 1)
        # project style images into clip embedding     
        emb_cross = self.control_cond_model(cond_cross)
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, 
                                         hint=cond_concat,
                                         timesteps=t, 
                                         context=emb_cross)

            def mask_control(c, mask_enable):
                if mask_enable:
                    resized_mask = T.Resize(list(c.shape[-2:]), T.InterpolationMode.NEAREST)(cond_mask)
                    #return c * resized_mask.unsqueeze(1)
                    return c * resized_mask
                else: 
                    return c
                
            control = [mask_control(c, mask_enable) * scale for c, scale, mask_enable in zip(control, self.control_scales, self.mask_enables)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=20, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, c_text, mask = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["c_text"][0][:N], c["c_concat_mask"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        reconstructed = self.decode_first_stage(z)
        #log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((64, 64), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, cartoon'

        cond = {"c_concat": [c_cat], 
                "c_crossattn": [c],
                "c_text": [c_text],
                'c_concat_mask': [mask]}

        un_cond = {"c_concat": [c_cat], 
                "c_crossattn": [torch.zeros_like(c)],
                "c_text":[self.get_learned_conditioning([n_prompt] * N)],
                'c_concat_mask': [mask] }
        
        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=cond,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            
            #uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            #uc_full = {"c_concat": [uc_cat], "c_text": [uc_cross], "c_crossattn": [torch.zeros_like(c)], 'c_concat_mask':[mask]}
            samples_cfg, _ = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=un_cond,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            #log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            log['concat'] = torch.cat((reconstructed, x_samples_cfg), dim=-2)

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # create folder
        f_ext = 'png' 
        sample_root = Path(self.logger.save_dir)/'samples'
        gt_root = Path(self.logger.save_dir)/'gt'
        #mask_root = Path(self.logger.save_dir)/'mask'
        src_root = Path(self.logger.save_dir)/'src'
        concat_root = Path(self.logger.save_dir)/'concat'

        for root_name in [sample_root, gt_root, src_root, concat_root]:
            os.makedirs(str(root_name), exist_ok=True)        
        # inference
        images = self.log_images(batch, N=len(batch), ddim_steps=20, unconditional_guidance_scale=9.0)
        
        images['samples'] = torch.clamp(images['samples'].detach().cpu() * 0.5 + 0.5, 0., 1.1)
        images['samples']/=(torch.max(torch.abs(images['samples'])))

        for k in ['src_img', 'jpg']:
            batch[k] = (rearrange(batch[k],'b h w c -> b c h w' ) + 1.0) / 2.0

        # save ground truth, source, mask
        # save samples
        for  sample, fname, src_image, gt in \
            zip(images['samples'], batch['fname'], batch['src_img'], batch['jpg']):

            #resized_mask = T.Resize(list(sample.shape[-2:]), T.InterpolationMode.NEAREST)(mask).to(sample.device)
            #sample *= resized_mask

            #neg_mask = (~resized_mask.bool())*torch.tensor(1.0, dtype=torch.float32).to(sample.device)
            #sample += neg_mask * T.Resize(sample.shape[-2:])(bg).to(sample.device)
            
            sample = T.CenterCrop(size=(512, 352))(sample)
            gt = T.CenterCrop(size=(512, 352))(gt)
            src_image = T.CenterCrop(size=(512, 352))(src_image)
            concat = T.Resize([256, 528])(torch.cat([src_image.detach().cpu(),
                                                     gt.detach().cpu(),
                                                     sample.detach().cpu()], 2))

            T.ToPILImage()(concat).save(concat_root/f'{fname}.{f_ext}')
        
            T.ToPILImage()(sample).save(sample_root/f'{fname}.{f_ext}')
            T.ToPILImage()(src_image).save(src_root/f'{fname}.{f_ext}')
            T.ToPILImage()(gt).save(gt_root/f'{fname}.{f_ext}')
            #T.ToPILImage()(mask).save(mask_root/f'{fname}.{f_ext}')

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.control_cond_model.proj.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.to(self.device)
            self.control_model = self.control_model.to(self.device)
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.to(self.device)
            self.cond_stage_model = self.cond_stage_model.to(self.device)

