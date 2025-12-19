import torch
import cv2
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dir(path, start, end):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, str(i) + '.jpg'))
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).to(device)
    return lmss, imgs_paths
