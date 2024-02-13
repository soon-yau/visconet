from share import *
import config
import os
import einops
import gradio as gr
import numpy as np
import torch
import random
import re
from datetime import datetime
from glob import glob
import argparse

from pytorch_lightning import seed_everything
from torchvision.transforms import ToPILImage
from annotator.util import pad_image, resize_image, HWC3
from annotator.openpose import  OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, log_txt_as_img
from visconet.segm import ATRSegmentCropper as SegmentCropper
from visconet.deepfashion import DeepFashionDatasetNumpy
from huggingface_hub import snapshot_download


# supply  directory of visual prompt images
HF_REPO = 'soonyau/visconet'
GALLERY_PATH = Path('./fashion/')
WOMEN_GALLERY_PATH = GALLERY_PATH/'WOMEN'
MEN_GALLERY_PATH = GALLERY_PATH/'MEN'

DEMO = True
LOG_SAMPLES = True
APP_FILES_PATH = Path('./app_files')
VISCON_IMAGE_PATH = APP_FILES_PATH/'default_images'
LOG_PATH = APP_FILES_PATH/'logs'
SAMPLE_IMAGE_PATH = APP_FILES_PATH/'samples'

DEFAULT_CONTROL_SCALE = 1.0
SCALE_CONFIG = {
    'Default': [DEFAULT_CONTROL_SCALE]*13, 
    'DeepFakes':[1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0,
                 0.5, 0.5, 0.5,
                 0.0, 0.0, 0.0, 0.0,],    
    'Faithful':[1,1,1,
                 1,1,1,
                 1,1,0.5,
                 0.5,0.5,0,0],
    'Painting':[0.0,0.0,0.0,
                0.5,0.5,0.5,
                0.5,0.5,0.5,
                0.5,0,0,0],
    'Pose':    [0.0,0.0,0.0,
                0.0,0.0,0.0,
                0.0,0.0,0.5,
                0.0,0.0,0,0],
    'Texture Transfer':  [1.0,1.0,1.0,
                1.0,1.0,1.0,
                0.5,0.0,0.5,
                0.0,0.0,0,0]
    }
DEFAULT_SCALE_CONFIG = 'Default'
ignore_style_list = ['headwear', 'accesories', 'shoes']

global device
global segmentor
global apply_openpose
global style_encoder
global model
global ddim_sampler
global dataset


def fetch_deepfashion(deepfashion_names):
    sample = dataset.get(deepfashion_names)
    input_image = sample['source_image']
    pose_image = sample['pose_image']
    mask_image = sample['mask_image']
    viscon_images = [sample['viscon_image'][style_name] for style_name in style_names]
    return [input_image, pose_image, mask_image, *viscon_images]

def select_gallery_image(evt: gr.SelectData):
    return evt.target.value[evt.index]['name']

def select_default_strength(strength_config):
    return SCALE_CONFIG[strength_config]

def change_all_scales(scale):
    return [float(scale)]*13

def encode_style_images(style_images):
    style_embeddings = []

    for style_name, style_image in zip(style_names, style_images):
        if style_image == None:
            style_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            
        #style_image = style_image.resize((224,224))            
        style_image = style_encoder.preprocess(style_image).to(device)
        style_emb = style_encoder.postprocess(style_encoder(style_image)[0])
        style_embeddings.append(style_emb)

    styles = torch.tensor(np.array(style_embeddings)).squeeze(-2).unsqueeze(0).float().to(device)
    return styles

def save_viscon_images(*viscon_images):
    ret_images = []
    for image, name in zip(viscon_images, style_names):
        fname = str(VISCON_IMAGE_PATH/name)+'.jpg'
        if image:
            image = image.resize((224,224))
            if os.path.exists(fname):
                os.remove(fname)
            image.save(fname)            
        ret_images.append(image)
    return ret_images


def extract_pose_mask(input_image, detect_resolution,
                      ignore_head=True, ignore_hair=False):
    # skeleton
    input_image = pad_image(input_image, min_aspect_ratio=0.625)
    detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution), hand=True)
    detected_map = HWC3(detected_map)
   
    # human mask
    cropped = segmentor(input_image, ignore_head=ignore_head, ignore_hair=ignore_hair)
    mask = cropped['human_mask']
    mask = Image.fromarray(np.array(mask*255, dtype=np.uint8), mode='L')

    return [detected_map, mask]

def extract_fashion(input_image):
    
    # style images
    cropped = segmentor(input_image)
    cropped_images = []
    for style_name in style_names:
        if style_name in cropped and style_name not in ignore_style_list:
            cropped_images.append(cropped[style_name])
        else:
            cropped_images.append(None)
    
    return [*cropped_images]

def get_image_files(image_path, ret_image=True, exts=['.jpg','.jpeg','.png']):
    images = []
    for ext in exts:
        images += [x for x in glob(str(Path(image_path)/f'*{ext}'))]
    if ret_image:
        images = [Image.open(x) for x in images]
    return images
                                                   
def log_sample(seed, results, prompt, skeleton_image,  mask_image, control_scales, *viscon_images):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = LOG_PATH/time_str
    os.makedirs(str(log_dir), exist_ok=True)

    # save result
    concat = np.hstack((skeleton_image, *results))
    Image.fromarray(skeleton_image).save(str(log_dir/'skeleton.jpg'))   
    Image.fromarray(mask_image).save(str(log_dir/'mask.png'))
    for i, result in enumerate(results):
        Image.fromarray(result).save(str(log_dir/f'result_{i}.jpg'))

    # save text
    with open(str(log_dir/'info.txt'),'w') as f:
        f.write(f'prompt: {prompt} \n')
        f.write(f'seed: {seed}\n')
        control_str = [str(x) for x in control_scales]
        f.write(','.join(control_str) + '\n')
    # save vison images
    for style_name, style_image in zip(style_names, viscon_images):
        if style_image is not None:
            style_image.save(str(log_dir/f'{style_name}.jpg'))


def process(prompt, a_prompt, n_prompt, num_samples,
            ddim_steps, scale, seed, eta, mask_image, pose_image,  
            c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
            *viscon_images):

    with torch.no_grad():
        control_scales = [c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0]
        mask = torch.tensor(mask_image.mean(-1)/255.,dtype=torch.float) #(512,512), [0,1]
        mask = mask.unsqueeze(0).to(device) # (1, 512, 512)
        style_emb = encode_style_images(viscon_images)

        # fix me
        detected_map = HWC3(pose_image)
        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        H, W, C = detected_map.shape
        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        new_style_shape = [num_samples] + [1] * (len(style_emb.shape)-1)
  
        cond = {"c_concat": [control], 
                "c_crossattn": [style_emb.repeat(new_style_shape)],
                "c_text": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
                'c_concat_mask': [mask.repeat(num_samples, 1, 1, 1)]}

        un_cond = {"c_concat": [control], 
                   "c_crossattn": [torch.zeros_like(style_emb).repeat(new_style_shape)],
                   "c_text":[model.get_learned_conditioning([n_prompt] * num_samples)],
                   'c_concat_mask': [torch.zeros_like(mask).repeat(num_samples, 1, 1, 1)]}
        
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = control_scales

        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    if LOG_SAMPLES:
        log_sample(seed, results, prompt, detected_map, mask_image, control_scales, *viscon_images)
    return results

def get_image(name, file_ext='.jpg'):
    fname = str(VISCON_IMAGE_PATH/name)+file_ext
    if not os.path.exists(fname):
        return None
    return Image.open(fname)
    
def get_image_numpy(name, file_ext='.png'):
    fname = str(VISCON_IMAGE_PATH/name)+file_ext
    if not os.path.exists(fname):
        return None
    return np.array(Image.open(fname))
    
def create_app():    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## ViscoNet: Visual ControlNet with Human Pose and Fashion <br> [Video tutorial](https://youtu.be/85NyIuLeV00)")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Get pose and mask", open=True):
                    with gr.Row():
                        input_image = gr.Image(source='upload', type="numpy", label='input image', value=np.array(get_image_numpy('ref')))
                        pose_image = gr.Image(source='upload', type="numpy", label='pose', value=np.array(get_image_numpy('pose')))
                        mask_image = gr.Image(source='upload', type="numpy", label='mask', value=np.array(get_image_numpy('mask')))
                    with gr.Accordion("Human Pose Samples", open=False):
                        with gr.Tab('Female'):
                            samples = get_image_files(str(SAMPLE_IMAGE_PATH/'pose/WOMEN/'))
                            female_pose_gallery = gr.Gallery(label='pose', show_label=False, value=samples).style(grid=3, height='auto')
                        with gr.Tab('Male'):
                            samples = get_image_files(str(SAMPLE_IMAGE_PATH/'pose/MEN/'))
                            male_pose_gallery = gr.Gallery(label='pose', show_label=False, value=samples).style(grid=3, height='auto')                        
                    with gr.Row():
                        #pad_checkbox = gr.Checkbox(label='Pad pose to square', value=True)
                        ignorehead_checkbox = gr.Checkbox(label='Ignore face in masking (for faceswap with text)', value=False)
                        ignorehair_checkbox = gr.Checkbox(label='Ignore hair in masking', value=False, visible=True)
                    with gr.Row():
                        #ignore_head_checkbox = gr.Checkbox(label='Ignore head', value=False)
                        get_pose_button = gr.Button(label="Get pose", value='Get pose')
                        get_fashion_button = gr.Button(label="Get visual", value='Get visual prompt')
                            
                
                with gr.Accordion("Visual Conditions", open=True):
                    gr.Markdown('Drag-and-drop, or click from samples below.')
                    with gr.Column():
                        viscon_images = []
                        viscon_images_names2index = {}
                        viscon_len = len(style_names)
                        v_idx = 0
                        
                        with gr.Row():
                            for _ in range(len(style_names)):
                                viscon_name = style_names[v_idx]
                                vis = False if viscon_name in ignore_style_list else True
                                viscon_images.append(gr.Image(source='upload', type="pil", min_height=112, min_width=112, label=viscon_name, value=get_image(viscon_name), visible=vis))
                                viscon_images_names2index[viscon_name] = v_idx
                                v_idx += 1

                        viscon_button = gr.Button(value='Save as Default',visible=False if DEMO else True)     

                    viscon_galleries = []

                    with gr.Accordion("Virtual Try-on", open=False):
                        with gr.Column():
                            #with gr.Accordion("Female", open=False):
                            with gr.Tab('Female'):
                                for garment, number in zip(['face', 'hair', 'top', 'bottom', 'outer'], [50, 150, 500, 500, 250]):
                                    with gr.Tab(garment):
                                        samples = []
                                        if WOMEN_GALLERY_PATH and os.path.exists(WOMEN_GALLERY_PATH):
                                            samples = glob(os.path.join(WOMEN_GALLERY_PATH, f'**/{garment}.jpg'), recursive=True)
                                            samples = random.choices(samples, k=number)
                                        viscon_gallery = gr.Gallery(label='hair', allow_preview=False, show_label=False, value=samples).style(grid=4, height='auto')
                                        viscon_galleries.append({'component':viscon_gallery, 'inputs':[garment]})
                            #with gr.Accordion("Male", open=False):
                            with gr.Tab('Male'):
                                for garment, number in zip(['face','hair', 'top', 'bottom', 'outer'], [50, 150, 500, 500, 250]):
                                    with gr.Tab(garment):
                                        samples = []
                                        if MEN_GALLERY_PATH and os.path.exists(MEN_GALLERY_PATH):
                                            samples = glob(os.path.join(MEN_GALLERY_PATH, f'**/{garment}.jpg'), recursive=True)
                                            samples = random.choices(samples, k=number)
                                        viscon_gallery = gr.Gallery(label='hair', allow_preview=False, show_label=False, value=samples).style(grid=4, height='auto')
                                        viscon_galleries.append({'component':viscon_gallery, 'inputs':[garment]})

            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, show_download_button=True, elem_id="gallery").style(grid=1, height='auto')
                with gr.Row():
                    max_samples = 8 if not DEMO else 4
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=max_samples, value=1, step=1)
                    scale_all = gr.Slider(label=f'Control Strength', minimum=0, maximum=1, value=DEFAULT_CONTROL_SCALE, step=0.05)
                    seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=2147483647, step=1, value=1561194236)#randomize=True) #value=1561194234)                
                    if not DEMO:
                        DF_DEMO = 'fashionWOMENTees_Tanksid0000762403_1front___fashionWOMENTees_Tanksid0000762403_1front'
                        DF_EVAL = 'fashionWOMENBlouses_Shirtsid0000035501_1front___fashionWOMENBlouses_Shirtsid0000035501_1front'
                        DF_RESULT ="fashionWOMENTees_Tanksid0000796209_1front___fashionWOMENTees_Tanksid0000796209_2side"                    
                        deepfashion_names = gr.Textbox(label='Deepfashion name', value=DF_EVAL)
                gr.Markdown("Default config reconstruct image faithful to pose, mask and visual condition. Reduce control strength to tip balance towards text prompt for more creativity.")
                prompt = gr.Textbox(label="Text Prompt", value="")
                
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    with gr.Accordion("Control Strength Scaling", open=False):
                        gr.Markdown("smaller value for stronger textual influence. c12 is highest spatial resolution controlling textures")
                        strength_select = gr.Dropdown(list(SCALE_CONFIG.keys()), label='strength settings', value=DEFAULT_SCALE_CONFIG)                        
                        scale_values = SCALE_CONFIG[DEFAULT_SCALE_CONFIG]
                        control_scales = []
                        c_idx = 12
                        with gr.Accordion("Advanced settings", open=False):
                            with gr.Row():
                                for _ in range(3):
                                    control_scales.append(gr.Slider(label=f'c{c_idx}', minimum=0, maximum=1, value=scale_values[12-c_idx], step=0.05))
                                    c_idx -= 1
                            with gr.Row():
                                for _ in range(3):
                                    control_scales.append(gr.Slider(label=f'c{c_idx}', minimum=0, maximum=1, value=scale_values[12-c_idx], step=0.05))
                                    c_idx -= 1
                            with gr.Row():
                                for _ in range(3):
                                    control_scales.append(gr.Slider(label=f'c{c_idx}', minimum=0, maximum=1, value=scale_values[12-c_idx], step=0.05))
                                    c_idx -= 1
                            with gr.Row():
                                for _ in range(4):
                                    control_scales.append(gr.Slider(label=f'c{c_idx}', minimum=0, maximum=1, value=scale_values[12-c_idx], step=0.05))
                                    c_idx -= 1                    
                    with gr.Row():
                        detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=512, value=512, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=12.0, step=0.1)

                    eta = gr.Number(label="eta (DDIM)", value=0.0, visible=False)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                        value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, sunglasses, hat')

        female_pose_gallery.select(fn=select_gallery_image, inputs=None, outputs=input_image)
        male_pose_gallery.select(fn=select_gallery_image, inputs=None, outputs=input_image)
        for vision_gallery in viscon_galleries:
            viscon_idx = viscon_images_names2index[vision_gallery['inputs'][0]]
            vision_gallery['component'].select(fn=select_gallery_image, inputs=None, 
                                            outputs=viscon_images[viscon_idx])
        ips = [prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, mask_image, pose_image, 
            *control_scales, *viscon_images]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
        prompt.submit(fn=process, inputs=ips, outputs=[result_gallery])
        get_pose_button.click(fn=extract_pose_mask, inputs=[input_image, detect_resolution, 
                                                            ignorehead_checkbox, ignorehair_checkbox], 
                            outputs=[pose_image, mask_image])
        get_fashion_button.click(fn=extract_fashion, inputs=input_image, outputs=[*viscon_images])    
        viscon_button.click(fn=save_viscon_images, inputs=[*viscon_images], outputs=[*viscon_images])
        strength_select.select(fn=select_default_strength, inputs=[strength_select], outputs=[*control_scales])
        scale_all.release(fn=change_all_scales, inputs=[scale_all], outputs=[*control_scales])
        if not DEMO:
            deepfashion_names.submit(fn=fetch_deepfashion, inputs=[deepfashion_names], 
                                     outputs=[input_image, pose_image, mask_image, *viscon_images])
    return block
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--config', type=str, default='./configs/visconet_v1.yaml')
    parser.add_argument('--ckpt', type=str, default='./models/visconet_v1.pth')
    parser.add_argument('--public_link', action='store_true', default='', help='Create public link')
    args = parser.parse_args()

    global device
    global segmentor
    global apply_openpose
    global style_encoder
    global model
    global ddim_sampler    
    global dataset
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    config_file = args.config
    model_ckpt = args.ckpt

    proj_config = OmegaConf.load(config_file)
    style_names = proj_config.dataset.train.params.style_names
    data_root = Path(proj_config.dataset.train.params.image_root)
    image_root = data_root/proj_config.dataset.train.params.image_dir
    style_root = data_root/proj_config.dataset.train.params.style_dir
    pose_root = data_root/proj_config.dataset.train.params.pose_dir
    mask_root = data_root/proj_config.dataset.train.params.mask_dir

    dataset = DeepFashionDatasetNumpy(**proj_config.dataset.val.params)
    segmentor = SegmentCropper()
    apply_openpose = OpenposeDetector()

    if not os.path.exists(model_ckpt):
        snapshot_download(repo_id=HF_REPO, local_dir='./models',
                        allow_patterns=os.path.basename(model_ckpt))

    style_encoder = instantiate_from_config(proj_config.model.style_embedding_config).to(device)
    model = create_model(config_file).cpu()    
    model.load_state_dict(load_state_dict(model_ckpt, location=device))

    model = model.to(device)
    model.cond_stage_model.device = device
    ddim_sampler = DDIMSampler(model)

    if not GALLERY_PATH.exists():
        zip_name = 'fashion.zip'
        snapshot_download(repo_id=HF_REPO, allow_patterns=zip_name, local_dir='.')
        from zipfile import ZipFile
        with ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_name)
    
    # Calling the main function with parsed arguments
    block = create_app()
    block.launch(share=args.public_link)
