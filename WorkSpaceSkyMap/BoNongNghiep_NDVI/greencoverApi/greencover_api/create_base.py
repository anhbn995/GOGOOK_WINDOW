from logging import exception
import os
import glob
import shutil

from all_step import step_1, step_2
from workspace import check_exists_foler
from mosaic.mosaic_image import main as merge_mosaic

def crop_image(img_path, box_path, tmp_path, all_path):
    out_cut= step_1(img_path, box_path, tmp_path)
    for i in glob.glob(out_cut+'/*.tif'):
        destination = os.path.join(all_path, os.path.basename(i))
        shutil.move(i, destination)
    shutil.rmtree(out_cut)
    return all_path


def main(folder_paths, temp_dir, box_path, weight_path_cloud):
    base_path = None
    type_img = '/*.tif'
    name = 'base_image_2020'
    temp_folder = os.path.join(temp_dir,'temp')
    tmp_path = os.path.join(folder_paths,'tmp')
    all_path = os.path.join(tmp_path,'all_img')
    out_cloud = os.path.join(tmp_path,'all')
    out_cloud_remove = os.path.join(out_cloud,'data_genorator_01/predict_float/cloud')
    cloud_tmp_dir = os.path.join(temp_dir,'mosaic')
    new_folder = os.path.join(folder_paths,'base_img')

    check_exists_foler(all_path)
    check_exists_foler(tmp_path)
    check_exists_foler(new_folder)
    check_exists_foler(temp_folder)
    check_exists_foler(cloud_tmp_dir)

    list_month = sorted(glob.glob(os.path.join(folder_paths,'T*')))

    for img_path in list_month:
        if os.path.exists(out_cloud):
            if os.path.exists(all_path):
                if len(glob.glob(out_cloud_remove+type_img))==len(glob.glob(all_path+type_img)):
                    print("Exists crop image and results of cloud remove.")
                else:
                    print("Crop folder exists but it's empty.")
                    print("Run crop image.")
                    crop_image(img_path, box_path, tmp_path, all_path, weight_path_cloud)
            else:
                print("Crop folder isn't exists.")
                print("Run crop image.")
                crop_image(img_path, box_path, tmp_path, all_path, weight_path_cloud)
        else:
            if os.path.exists(all_path):
                if len(glob.glob(all_path+type_img))==len(glob.glob(os.path.join(folder_paths,'T*'+type_img))):
                    print("Exists crop image but cloud folder isn't exists.")
                    print("Run cloud remove.")
                    out_cloud_tmp = step_2(all_path, tmp_path, weight_path_cloud)
                else:
                    print("Crop folder exists but it's empty.")
                    print("Run crop image.")
                    crop_image(img_path, box_path, tmp_path, all_path, weight_path_cloud)
            else:
                print("Crop folder and cloud folder aren't exists.")
                print("Run crop image.")
                crop_image(img_path, box_path, tmp_path, all_path, weight_path_cloud)
    
    print("Run cloud remove and mosaic.")
    out_cloud_tmp = step_2(all_path, tmp_path, weight_path_cloud)
    out_merge_mosaic = merge_mosaic(out_cloud_tmp, cloud_tmp_dir, temp_folder, base_path, name)
    out_img = os.path.join(new_folder, os.path.basename(out_merge_mosaic))
    shutil.copyfile(out_merge_mosaic, out_img)

    try:
        shutil.rmtree(tmp_path)
        shutil.rmtree(temp_folder)
        shutil.rmtree(cloud_tmp_dir)
    except:
        raise Exception("Can't clean workspace.")
    return new_folder

if __name__=="__main__":
    temp_dir = '/home/geoai/geoai_data_test'
    weight_path_cloud = '/home/quyet/WorkSpace/Quyet/greencover/weights/cloud_weights.h5'
    folder_paths = '/home/nghipham/Desktop/Jupyter/data/DA/4_CLRM/Sentinel2_workspace/2020'
    box_path = '/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/Green Cover Npark Singapore/box'
    
    main(folder_paths, temp_dir, box_path, weight_path_cloud)
