import os
import json
import glob
import shutil
from utils import write_shp, get_crs

def check_exists_foler(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def check_and_remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)
    return folder_path

def init_workspace(folder_paths, temp_dir, box_aoi, crs):
    if not os.path.exists(folder_paths):
        raise Exception("Input folder isn't exists, please check %s"%(folder_paths))

    if not os.path.exists(temp_dir):
        raise Exception("Temp folder isn't exists, please check %s"%(temp_dir))
        
    box_path = os.path.join(folder_paths, 'box')
    tmp_path = os.path.join(folder_paths, 'tmp')
    base_path = os.path.join(folder_paths,'base')
    results_path = os.path.join(folder_paths,'results')
    tmp_path = check_exists_foler(tmp_path)
    box_path = check_exists_foler(box_path)
    base_path = check_exists_foler(base_path)
    results_path = check_exists_foler(results_path)

    write_shp(box_aoi, box_path, os.path.basename(folder_paths)+'_AOI', crs)
    temp_folder = os.path.join(temp_dir,'temp')
    cloud_tmp_dir = os.path.join(temp_dir,'mosaic')
    temp_folder = check_and_remove_folder(temp_folder)
    cloud_tmp_dir = check_and_remove_folder(cloud_tmp_dir)
    return box_path, tmp_path, base_path, results_path, temp_folder, cloud_tmp_dir

def check_list_download_image(folder_paths, list_all_img):
    for i in list_all_img:
        list_img_check = list_all_img['%s'%str(i)]
        for j in list_img_check:
            month_check = 'T%s'%str(i)
            # print(list_all_img)
            path_check = os.path.join(folder_paths, month_check, j)
            if os.path.exists(path_check):
                pass
            else:
                raise Exception("Image file isn't exists, please check %s"%(path_check))
    return True


def init_input_fn(debug):
    json_path = os.path.join(os.getcwd(), 'requirements.json')
    if os.path.exists(json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
    else:
        raise Exception("Requirement file isn't exists, please check %s"%(json_path))

    if debug:
        static_result = False
        temp_dir = '/home/geoai/geoai_data_test'
        folder_paths = '/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/Kolkata/aaa'
        weight_path_cloud = '/home/quyet/WorkSpace/Quyet/greencover/weights/cloud_weights.h5'
        weight_path_green = '/home/quyet/WorkSpace/Quyet/greencover/weights/green_weights.h5'
        weight_path_water = '/home/quyet/WorkSpace/Quyet/greencover/weights/water_weights.h5'

    else:
        static_result = False
        temp_dir = data['temp_path']
        folder_paths = data['workspace_path']
        weight_path_cloud = data['weights']['cloud']
        weight_path_green = data['weights']['green']
        weight_path_water = data['weights']['water']
        list_all_img = data['list_image']
        box_aoi = data['AOI']
        check_list_download_image(folder_paths, list_all_img)

        # box_path = os.path.join(folder_paths, 'box')
        base_path = os.path.join(folder_paths,'base')
        list_month = sorted(glob.glob(os.path.join(folder_paths,'T*')))

        crs = get_crs(folder_paths, list_month)
        # write_shp(box_aoi, box_path, os.path.basename(folder_paths)+'_AOI', crs)
        if len(list_all_img.keys())==1:
            file_base = glob.glob(os.path.join(base_path,'*.tif'))
            if not file_base:
                raise Exception("If only run with 1 month, you need add base image")
    
    return static_result, temp_dir, folder_paths, weight_path_cloud, weight_path_green, weight_path_water, box_aoi, list_month, crs
    
