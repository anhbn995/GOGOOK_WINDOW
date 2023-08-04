#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 4 23:17:10 2021

@author: ducanh
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
import gdalnumeric


def get_list_name_file(path_folder, name_file = '*.tif'):
    """
        Get all file path with file type is name_file.
    """
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        head, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir


def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=data.dtype,
                        crs=crs,
                        transform=tr,
                        nodata=0,
                        ) as dst:
                            dst.write(data)


def get_index_cloud_for_4band(path_mask_cloud):
    """
        get anotation cloud
    """
    src_mask = rasterio.open(path_mask_cloud)
    img_4band = np.empty((4, src_mask.height, src_mask.width))
    for i in range(4):
        img_4band[i] = src_mask.read(1)
    index_cloud = np.where(img_4band == 255)
    return index_cloud


def export_img_cloud_to_nodata(img_path, mask_cloud_path, out_path):
    """
        set cloud is nodata
    """
    # get index_cloud
    index_cloud = get_index_cloud_for_4band(mask_cloud_path)

    # Set nodata
    src = rasterio.open(img_path)
    img = src.read()
    img[index_cloud] = 0
    write_image(img, src.height, src.width, src.count, src.crs, src.transform, out_path)


def mosaic(dir_path, list_img_name, out_path, base_img=None):
    src_files_to_mosaic = []
    # print(list_img_name)
    for name_f in list_img_name:
        fp = os.path.join(dir_path, name_f)
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    if base_img:
        src_files_to_mosaic.append(rasterio.open(base_img))
    mosaic, out_trans = merge(src_files_to_mosaic)
    write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)


def sort_list_file_by_cloud(dir_predict):
    list_fname = get_list_name_file(dir_predict)
    dict_name = dict.fromkeys(list_fname)
    for fname in list_fname:
        fp = os.path.join(dir_predict, fname)
        raster_file = gdalnumeric.LoadFile(fp)
        count = (raster_file==255).sum()
        dict_name[fname]=count
    dict_name_sort = sorted(dict_name.items(), key=lambda x: x[1])
    list_sort_name = list(dict(dict_name_sort).keys())
    return list_sort_name

def sort_list_file_by_date(list_fp_img_selected):
    list_sort_name = []
    for fp in list_fp_img_selected:
        name = os.path.basename(fp)
        list_sort_name.append(name)
    return list_sort_name


def main_mosaic(list_fp_img_selected, dir_predict_float, out_fp_cloud_remove, sort_amount_of_clouds, first_image, base_image=None):
    out_cloud = os.path.join(dir_predict_float, "cloud")
    if not os.path.exists(out_cloud):
        os.makedirs(out_cloud)
    if sort_amount_of_clouds:
        list_fn_sort = sort_list_file_by_cloud(dir_predict_float)
    else:
        list_fn_sort = sort_list_file_by_date(list_fp_img_selected)

    # if base_image:
    #     list_fp_img_selected.append(base_image)

    if first_image:
        name = os.path.basename(first_image)
        list_fn_sort.remove(name)
        list_fn_sort.insert(0, name)
        
    for fp in list_fp_img_selected:
        fname = os.path.basename(fp)
        fp_mask = os.path.join(dir_predict_float, fname)
        fp_cloud_rm = os.path.join(out_cloud, fname)
        export_img_cloud_to_nodata(fp, fp_mask, fp_cloud_rm)
        
    print(list_fn_sort)
    mosaic(out_cloud, list_fn_sort, out_fp_cloud_remove, base_image)
    return out_cloud
    
# xong xoa
# list_img_name = []
# for i in range(9, 0, -1):
#     list_img_name.append(f"T{i}.tif")
# dir_path = r"/home/skm/SKM_OLD/public/DA/4_CLRM/Sentinel2_workspace/2020_crop_aoi_singapor_CLRM"
# out_path = r"/home/skm/SKM_OLD/public/DA/4_CLRM/Sentinel2_workspace/2020_crop_aoi_singapor_CLRM/mosaic/v4_bo10.tif"    
# mosaic(dir_path, list_img_name, out_path)



# predict_float = r"/home/skm/SKM/WORK/Cloud_and_mosaic/Panama/img_origin_cut_img_convert_01_float/predict_model_float"
# list_img_name = sort_list_file_by_cloud(predict_float)



# folder_img = r"/home/skm/SKM/WORK/Cloud_and_mosaic/Panama/Demo_Panama/img_sentinel2"
# folder_mask = r"/home/skm/SKM/WORK/Cloud_and_mosaic/Panama/Demo_Panama/img_sentinel2_convert_01_float_predict_model_float"
# out_path = r"/home/skm/SKM/WORK/Cloud_and_mosaic/Panama/Demo_Panama/mosaic_v22222.tif"
# out_tmp_nodata_cloud = r"/home/skm/SKM/WORK/Cloud_and_mosaic/Panama/Demo_Panama/img_sentinel2_convert_01_float_predict_model_float/tmp"
# if not os.path.exists(out_tmp_nodata_cloud):
#     os.makedirs(out_tmp_nodata_cloud)


# folder_img = '/home/nghipham/Desktop/Jupyter/data/DA/5_India/Hyderabad/T1/a/tm/a_cut'
# folder_mask = 
# list_fn_img = get_list_name_file(folder_img)
# for fname in list_fn_img:
#     fp = os.path.join(folder_img, fname)
#     fp_mask = os.path.join(folder_mask, fname)
#     fp_nodata = os.path.join(out_tmp_nodata_cloud, fname)
#     export_img_cloud_to_nodata(fp, fp_mask, fp_nodata)

# mosaic(out_tmp_nodata_cloud, list_img_name,out_path)

# dir_path = r"/home/nghipham/Desktop/Jupyter/data/DA/5_India/Hyderabad/T1/a/tm/a_cut"
# list_img_name = get_list_name_file(dir_path)
# out_path = r"/home/nghipham/Desktop/Jupyter/data/DA/5_India/Hyderabad/T1/a/tm/a_cut/a.tif"
# mosaic(dir_path, list_img_name, out_path, base_img=None)