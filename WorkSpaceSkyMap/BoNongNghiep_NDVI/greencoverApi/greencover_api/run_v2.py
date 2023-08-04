from genericpath import exists
import os
import glob
import shutil
import argparse

from change_model import get_model
from check_results import main as check_result
from greencover import detect_green, detect_water
from greencover.merge_water_green import combine_all
from mosaic.mosaic_image import main as merge_mosaic
from cloud_remove.cloud_remove import main as cloud_remove
from download_crop.crop_image import main_cut_img as crop_image
from cloud_mask.cloud_remove_mask import intersection_cloud, intersection_cloud_2
from utils import convert_profile, standard_coord, get_list_fp, sorted_month_folder, get_crs,\
                 reproject_image, convert_profile, reproject_profile

def crop_image_month(input_path, tmp_path, img_path, box_path, name, crs):
    out_crop_dir = os.path.join(tmp_path, name+'_cut')
    if os.path.exists(out_crop_dir):
        # kiem tra so file trong thu muc crop co du ko 
        if len(glob.glob(out_crop_dir+'/*.tif')) == len(glob.glob(os.path.join(input_path, name, '*.tif'))): 
            print("Exist crop image folder %s and isn't empty"%(name+'_cut'))
        # neu khong du so luong file trong thu muc crop thi chay lai crop anh
        else:
            print("Exist crop image folder %s but is empty"%(name+'_cut'))
            print("Crop image %s..."%(name))
            out_crop_dir = crop_image(img_path, box_path, tmp_path)
            standard_coord(out_crop_dir, crs)
    # neu khong ton tai thu muc crop thi chay crop anh va sau do chay cloud remove
    else:   
        print("Crop image folder isn't exists")
        print("Crop image %s..."%(name))
        print("\n")
        out_crop_dir = crop_image(img_path, box_path, tmp_path)
        standard_coord(out_crop_dir, crs)
    return out_crop_dir

def cloud_remove_month(input_path, out_crop_dir, tmp_path, weight_path_cloud, base_image_gdal, name):
    if os.path.exists(os.path.join(tmp_path, name)):
        num_img = len(glob.glob(os.path.join(input_path,'*.tif')))
        num_predict = len(glob.glob(os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud','*.tif')))
        if num_img == num_predict:
            print("Exist cloud remove result %s"%(name))
            out_img_cloud = os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud')
        else:
            print("Cloud remove result folder %s is empty"%(name))
            print("Run cloud remove...")
            print("\n")
            list_fp_img_selected = get_list_fp(out_crop_dir)
            sort_amount_of_clouds = True
            first_image = None
            tmp_dir = os.path.join(tmp_path, name)
            out_fp = os.path.join(tmp_dir, name + ".tif")
            out_img_cloud = cloud_remove(list_fp_img_selected, tmp_dir, out_fp, weight_path_cloud, sort_amount_of_clouds, first_image, base_image_gdal)
    else:
        print("Cloud remove result folder %s is empty"%(name))
        print("Run cloud remove...")
        print("\n")
        list_fp_img_selected = get_list_fp(out_crop_dir)
        sort_amount_of_clouds = True
        first_image = None
        tmp_dir = os.path.join(tmp_path, name)
        out_fp = os.path.join(tmp_dir, name + ".tif")
        out_img_cloud = cloud_remove(list_fp_img_selected, tmp_dir, out_fp, weight_path_cloud, sort_amount_of_clouds, first_image, base_image_gdal)
    return out_img_cloud

def check_exists_foler(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def check_exists_foler_with_name(folder_path, name):
    out_folder = os.path.join(folder_path, name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    return out_folder

def check_input(list_month, box_path, base_path, cloud_tmp_dir, use_box=True):
    print("Start check input file(box, image, base) and remove old mosaic tmp ...")
    if os.path.exists(cloud_tmp_dir):
        shutil.rmtree(cloud_tmp_dir)

    if len(glob.glob(os.path.join(box_path, '*.shp')))==0 and use_box:
        raise Exception("Box folder is empty.")
    
    for i in list_month:
        if len(glob.glob(os.path.join(i,'*.tif'))) == 0:
                raise Exception("Image folder is empty")

    if glob.glob(os.path.join(base_path,'*.tif')):
        base_image_gdal = glob.glob(os.path.join(base_path,'*.tif'))[0]
        print("Base image is exists")
    else:
        print("Base image isn't exists")
        base_image_gdal = None
    return base_image_gdal

def mosaic_PCI(results_img, out_img_cloud, base_path, cloud_tmp_dir, temp_folder, name):
    check_exists_foler(cloud_tmp_dir)
    cloud_tmp_geotest= os.path.join(cloud_tmp_dir, name)
    if not os.path.exists(results_img):
        print("Merge image and mosaic with PCI...")
        out_merge_mosaic = merge_mosaic(out_img_cloud, cloud_tmp_geotest, temp_folder, base_path, name)
        shutil.copyfile(out_merge_mosaic, results_img)
        print("\n")
    return results_img

def mosaic_GDAL(folder_path, tmp_path, name):
    name_img = name+'_DA.tif'
    img_tmp = os.path.join(tmp_path, name, name+'.tif')
    folder_mosaic = os.path.join(folder_path, 'DA')
    check_exists_foler(folder_mosaic)
    img_mosaic = os.path.join(folder_mosaic, name_img)
    if not os.path.exists(img_mosaic):
        shutil.copyfile(img_tmp, img_mosaic)
    return img_mosaic, img_tmp

def reproject_PCI_GDAL(results_img, tmp_path, name, crs):
    print("Reproject base image")
    aaa = os.path.join(tmp_path, name, 'base_PCI')
    check_exists_foler(aaa)
    reproject_image(src_path=results_img, dst_path=os.path.join(aaa, os.path.basename(results_img)), dst_crs=crs)
    base_path = aaa

    bbb = os.path.join(tmp_path, name, 'base_DA')
    check_exists_foler(bbb)
    reproject_image(src_path=os.path.join(tmp_path, name, '%s.tif'%(name)), dst_path=os.path.join(bbb, os.path.basename(results_img)), dst_crs=crs)
    base_image_gdal = os.path.join(bbb, '%s.tif'%(name))
    print("\n")
    return base_path, base_image_gdal

def update_proflie_each_month(out_mosaic_PCI, out_mosaic_GDAL, profile_img_path_PCI, profile_img_path_DA, name):
    print("Update profile for image %s"%(name))
    # print(profile_img_path_PCI, results_img)
    # PCI_temp = reproject_profile(profile_img_path_PCI, results_img)
    img_temp_PCI = out_mosaic_PCI.replace('.tif', '_new.tif')
    convert_profile(out_mosaic_PCI, profile_img_path_PCI, img_temp_PCI)
    os.remove(out_mosaic_PCI)
    shutil.copyfile(img_temp_PCI, out_mosaic_PCI)
    os.remove(img_temp_PCI)

    GDAL_temp = reproject_profile(profile_img_path_DA, out_mosaic_GDAL)
    os.remove(out_mosaic_GDAL)
    shutil.copyfile(GDAL_temp, out_mosaic_GDAL)
    os.remove(GDAL_temp)
    print("\n")
    return True

def run_segmentation(img, result_path, weight_path_green, weight_path_water, dil=False, run_agian=False):
    green_model, input_size_green = get_model("Green_model")
    water_model, input_size_water = get_model("Water_model")
    if not os.path.exists(img.replace('.tif', '_green.tif')):
        print(img)
        tmp_green = detect_green.predict(img, result_path, weight_path_green, green_model, input_size_green, dil=dil)
    else:
        if run_agian:
            tmp_green = detect_green.predict(img, result_path, weight_path_green, green_model, input_size_green, dil=dil)
        else:
            tmp_green = None
            pass
    # tmp_green = detect_green.predict(img, result_path, weight_path_green, dil=False)
    if not os.path.exists(img.replace('.tif', '_water.tif')):
        print(img)
        tmp_water = detect_water.predict(img, result_path, weight_path_water, water_model, input_size_water)
    else:
        if run_agian:
            tmp_water = detect_water.predict(img, result_path, weight_path_water, water_model, input_size_water)
        else:
            tmp_water = None
            pass
    # tmp_water = detect_water.predict(img, result_path, weight_path_water)
    return tmp_green, tmp_water
    # return tmp_green

def classification(list_removecloud_img, img_mosai_gdal_tmp, folder_paths, weight_path_green, weight_path_water):
    for j in list_removecloud_img:
        # chay classification voi anh PCI
        print("******Classification greencover %s******"%(os.path.basename(j).split('.')[0]))
        new_folder_path = os.path.join(folder_paths, 'results', os.path.basename(j).split('.')[0])
        result_green, result_water = run_segmentation(j, new_folder_path, weight_path_green, weight_path_water)
        print("Colormap results")
        combine_all(j, result_green, result_water)

        # chay classification voi anh mosaic bang gdal
        print("Run classifiacation with gdal mosaic image")
        name_img = os.path.basename(j).replace('.tif', '_DA.tif')
        folder_mosaic = os.path.join(new_folder_path, 'DA')
        img_mosaic = os.path.join(folder_mosaic, name_img)

        if not os.path.exists(img_mosaic):
            shutil.copyfile(img_mosai_gdal_tmp, img_mosaic)
        result_green_tmp, result_water_tmp = run_segmentation(img_mosaic, folder_mosaic, weight_path_green, weight_path_water)
        combine_all(img_mosaic, result_green_tmp, result_water_tmp)
        print("\n")

def clean_workspace(temp_folder, cloud_tmp_dir):
    print("Clean workspace")
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    except:
        raise Exception("Can't remove tmp folder, please check it.")
    
    try:
        if os.path.exists(cloud_tmp_dir):
            shutil.rmtree(cloud_tmp_dir)
    except:
        raise Exception("Can't remove mosaic folder, please check it.")

def cloud_mask(folder_path, tmp_path, name):
    print("*Create cloud remove mask")
    # shp_temp_path = os.path.join(tmp_path, name)
    out_shp_path = os.path.join(folder_path, os.path.basename(folder_path)+'.shp')
    tif_folder_path = os.path.join(tmp_path, name, 'data_genorator_01/predict_float')
    if not os.path.exists(out_shp_path):
        intersection_cloud_2(tif_folder_path, out_shp_path)
    else:
        print("**Mask is exists")

    # if not os.path.exists(out_shp_path):
    #     intersection_cloud(tif_folder_path, shp_temp_path, out_shp_path)
    # else:
    #     print("**Mask is exists")
    return out_shp_path

def main():
    debug = True
    if not debug:
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--folder_dir', help='Path of folder workspace', required=False, type=str,
                                default='/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/Kolkata')
        args_parser.add_argument('--temp_dir', help='Path of folder contain output of pci mosaic', required=False, 
                                type=str, default='/home/geoai/geoai_data_test/temp')
        args_parser.add_argument('--cloud_tmp_dir', help='Path of folder contain image to pci mosaic', required=False, 
                                type=str, default='/home/geoai/geoai_data_test/mosaic')
        args_parser.add_argument('--weight_path_cloud', help='Path of cloud remove model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Greencover_api/greencover/weights/cloud_weights.h5')
        args_parser.add_argument('--weight_path_green', help='Path of green detect model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Greencover_api/greencover/weights/green_weights.h5')
        args_parser.add_argument('--weight_path_water', help='Path of water detect model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Greencover_api/greencover/weights/water_weights.h5')
        args_parser.add_argument('--check_result', help='Turn on/off check result and report', required=False, 
                                type=bool, default=True)
        param = args_parser.parse_args()
        static_result = param.check_result
        temp_folder = param.temp_dir
        cloud_tmp_dir = param.cloud_tmp_dir
        folder_paths = param.folder_dir
        weight_path_cloud = param.weight_path_cloud
        weight_path_green =param.weight_path_green
        weight_path_water = param.weight_path_water
    else:
        static_result = False
        temp_folder = '/home/geoai/geoai_data_test/temp'
        cloud_tmp_dir = '/home/geoai/geoai_data_test/mosaic'
        # thu muc chua cai folder chi theo thang ,vi du : T1, T2,...
        folder_paths = '/home/quyet/data/bk_2'
        weight_path_cloud = '/home/quyet/WorkSpace/Greencover_api/greencover/weights/cloud_weights.h5'
        weight_path_green = '/home/quyet/WorkSpace/Greencover_api/greencover/weights/green_weights.h5'
        weight_path_water = '/home/quyet/WorkSpace/Greencover_api/greencover/weights/water_weights.h5'
        # weight_path_water = '/home/nghipham/Desktop/Jupyter/data/data_greencover_sing/Weights/unet3pluswater_256_1class_binary_val.h5'

    use_box = False
    box_path = check_exists_foler_with_name(folder_paths, 'box')
    base_path_PCI = check_exists_foler_with_name(folder_paths,'base')
    tmp_path = check_exists_foler_with_name(folder_paths, 'tmp')
    results_path = check_exists_foler_with_name(folder_paths,'results')
    list_month = sorted_month_folder(folder_paths)
    crs = get_crs(folder_paths, list_month)
    base_image_gdal = check_input(list_month, box_path, base_path_PCI, cloud_tmp_dir, use_box)

    check_month=[]
    list_removecloud_img = []
    for n, i in enumerate(list_month):
        name = os.path.basename(i)
        check_month.append(os.path.basename(i))
        results_month_path = check_exists_foler_with_name(results_path, name)
        results_img = os.path.join(results_month_path, name+'.tif')
        print("Check results of colud remove")
        if not os.path.exists(results_img):
            print("Not exist")
            print("Check results old predict cloud:")
            # To do:
            if not use_box:
                out_crop_dir = os.path.join(folder_paths, i)
            else:
                out_crop_dir = crop_image_month(folder_paths, tmp_path, i, box_path, name, crs)

            out_img_cloud = cloud_remove_month(out_crop_dir, out_crop_dir, tmp_path, weight_path_cloud, base_image_gdal, name)
        else:
            out_img_cloud = os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud')
            print("Exist cloud remove result %s"%(name))    

        out_mosaic_PCI = mosaic_PCI(results_img, out_img_cloud, base_path_PCI, cloud_tmp_dir, temp_folder, name)
        out_mosaic_GDAL, img_mosai_gdal_tmp = mosaic_GDAL(results_month_path, tmp_path, name)
        base_path_PCI, base_image_gdal = reproject_PCI_GDAL(results_img, tmp_path, name, crs)
        if n == 0:
            print("*Get profile of image T1")
            profile_img_path_PCI = out_mosaic_PCI
            profile_img_path_DA = out_mosaic_GDAL
        else:
            update_proflie_each_month(out_mosaic_PCI, out_mosaic_GDAL, profile_img_path_PCI, profile_img_path_DA, name)
        print("******", os.path.basename(name),"finished******",)
        list_removecloud_img.append(results_img)
    
    classification(list_removecloud_img, img_mosai_gdal_tmp, folder_paths, weight_path_green, weight_path_water)
    clean_workspace(temp_folder, cloud_tmp_dir)
    if static_result:
        threshold = 0.1
        check_result(results_path, check_month, threshold)
    print("******Finished******")
    return True

if __name__=="__main__":
    main()