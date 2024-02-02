import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

from src import model
from src import util
from src.body import Body
from src.hand import Hand

from PIL import Image
from glob import glob
from tqdm import tqdm


body_estimation = Body('model/body_pose_model.pth')

src_dir = '/home/soon/datasets/deepfashion_inshop/img_512/'
dst_dir = '/home/soon/datasets/deepfashion_inshop/img_512_padded/'
skeleton_dir = '/home/soon/datasets/deepfashion_inshop/openpose_512/'
src_files = glob(src_dir+'**/*.jpg', recursive=True)


for src_file in tqdm(src_files[:]):
    src = cv2.imread(src_file)
    # add padding
    top = 0
    bottom = 0
    left = right = 82
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_REPLICATE)
    dst_file = src_file.replace(src_dir, dst_dir)
    os.makedirs(os.path.split(dst_file)[0], exist_ok=True)
    cv2.imwrite(dst_file, dst)
    # plt.imshow(dst[:,:,::-1])
    
    # skeleton
    candidate, subset = body_estimation(dst)
    canvas = copy.deepcopy(dst)
    canvas = util.draw_bodypose(np.zeros_like(canvas), candidate, subset)
    skeleton = Image.fromarray(canvas)    
    
    skeleton_file = src_file.replace(src_dir, skeleton_dir)
    os.makedirs(os.path.split(skeleton_file)[0], exist_ok=True)
    skeleton.save(skeleton_file)
