import sys
import os
import pdb
from Dataset4EO.datasets import GlobalBuildingRaster
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2


# in_root = '/home/Datasets/Dataset4EO/GlobalBF/so2sat/'
# in_base_dir = './'
# out_root = './outputs/global_bf_africa'

in_root = '/home/Datasets/Dataset4EO/GlobalBF/so2sat/planet_global_processing/Continents/EUROPE'
in_base_dir = 'glcv103_guf_wsf' # this prefix will be copied to the out_dir
out_root = '/home/Datasets/Dataset4EO/GlobalBF/ai4eo'

from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = GlobalBuildingRaster(
        root=in_root, in_base_dir=in_base_dir, out_root=out_root, split='train',
        crop_size=(10000, 10000), stride=(10000, 10000), patchify=True
    )
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    for i, it in enumerate(dp):
        print(f'iter {i}: {it}')
        if i == 100:
            break
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))
