import os
import glob
import shutil
import rasterio
import numpy as np

from mosaic.mosaic_image import check_base_img


def count_compare(img_cur_path, img_base_path):
    img_cur = rasterio.open(img_cur_path).read_masks(1)
    img_base = rasterio.open(img_base_path).read_masks(1)
    numpix_cur = np.sum(img_cur!=0)
    numpix_base = np.sum(img_base!=0)
    if numpix_base == numpix_cur:
        return True
    else:
        return False

def check_equal_img(result_path):
    max_pix = 0 
    status = True
    for i in sorted(result_path):
        img = rasterio.open(i).read_masks()
        cur_pix = np.sum(img!=0)
        if cur_pix > max_pix:
            max_pix = cur_pix
            status = False
            if status == False:
                break
    return status

def pick_base(result_path, base_path):
    max_pix = 0
    for i in sorted(result_path):
        img = rasterio.open(i).read()
        cur_pix = np.sum(img!=0)
        if cur_pix > max_pix:
            max_pix = cur_pix
            cur_path = i
    base_path_dir = os.path.join(base_path, os.path.basename(cur_path))
    shutil.copyfile(cur_path, base_path_dir)
    # base_path_dir is the link tiff image
    return base_path_dir

def add_base(img_cur_path, img_base_path):
    try:
        check_base_img(img_base_path)
    except:
        raise Exception("Not exist base image")
    check_imgs = count_compare(img_cur_path, img_base_path)
    if check_imgs:
        print("*Image equal base")
    else:
        print("*Image not equal base")

