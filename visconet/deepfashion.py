import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from abc import abstractmethod
import pickle
import numpy as np
from einops import rearrange


class Loader(Dataset):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, ind):
        pass

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

def list_subdirectories(path):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not dirnames:
            subdirectories.append(dirpath)
    return subdirectories
        
class DeepFashionDataset(Loader):
    def __init__(self,
                 image_root,
                 image_dir,
                 pose_dir,
                 style_dir,
                 mask_dir,
                 map_file,
                 data_files:list,
                 dropout=None,
                 sample_ratio=None,
                 style_postfix='',
                 image_shape=[512, 512],
                 style_emb_shape=[1, 768],
                 style_names=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.root = Path(image_root)
        self.image_root = self.root/image_dir
        self.pose_root = self.root/pose_dir
        self.style_root = self.root/style_dir
        self.mask_root = self.root/mask_dir
        self.map_df = pd.read_csv(map_file)
        self.map_df.set_index('image', inplace=True)
        dfs = [pd.read_csv(f) for f in data_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.style_postfix = style_postfix
        if sample_ratio:
            self.df = self.df.head(int(sample_ratio*len(self.df)))
        # transformation
        self.image_tform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        
        self.skeleton_tform = T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))])        
        
        self.clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = T.Compose([
            T.ToTensor(),
            self.clip_norm
        ])

        self.style_names = style_names
        self.style_emb_shape = style_emb_shape

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        try:
            row = self.df.iloc[index]
            fname = get_name(row['from'], row['to'])

            # source - get fashion styles

            source = self.map_df.loc[row['from']]
            src_path = str(self.image_root/source.name)
            source_image = self.image_tform(Image.open(src_path))
          
            styles_path = source['styles']
            if styles_path == np.nan:
                return self.skip_sample(index)
            
            full_styles_path = self.style_root/source['styles']

            style_embeddings = []
            for style_name in self.style_names:
                f_path = full_styles_path/(f'{style_name}'+self.style_postfix+'.p')
                if f_path.exists():
                    with open(f_path, 'rb') as file:
                        style_emb = pickle.load(file)
                else:
                    #style_emb = np.zeros((1,768), dtype=np.float16)
                    style_emb = np.zeros(self.style_emb_shape, dtype=np.float32)
                style_embeddings.append(style_emb)
            styles = torch.tensor(np.array(style_embeddings)).squeeze(-2)

            # target - get ground truth and pose
            target = self.map_df.loc[row['to']]           
            target_path = str(self.image_root/target.name)
            target_image = self.image_tform(Image.open(target_path))

            ## pose
            target_path = str(self.pose_root/target.name)
            pose_image = self.skeleton_tform(Image.open(target_path))

            prompt = 'a person'
            mask = T.ToTensor()(Image.open(str(self.mask_root/target.name).replace('.jpg','_mask.png')))
            
            return dict(jpg=target_image, 
                        txt=prompt, 
                        hint=pose_image,
                        styles=styles,
                        human_mask=mask,
                        src_img=source_image,
                        fname=fname)
        
        except Exception as e:            
            #print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)