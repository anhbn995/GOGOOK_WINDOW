import os
import glob
import shutil
import rasterio
import argparse

from check_results import main as check_result
from greencover.merge_water_green import combine_all
from all_step import step_1, step_2, step_3, step_4
from mosaic.mosaic_image import main as merge_mosaic


if __name__ == '__main__':
    debug = True
    if not debug:
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--folder_dir', help='Path of folder workspace', required=False, type=str,
                                default='/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/Kolkata')
        args_parser.add_argument('--temp_dir', help='Path of folder contain output of pci mosaic', required=False, 
                                type=str, default='/home/geoai/geoai_data_test/temp')
        args_parser.add_argument('--cloud_tmp_dir', help='Path of folder contain image to pci mosaic', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Model/swin_rotation/data_result')
        args_parser.add_argument('--weight_path_cloud', help='Path of cloud remove model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Quyet/greencover/weights/20210603_011610_0109_val_weights.h5')
        args_parser.add_argument('--weight_path_green', help='Path of green detect model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Quyet/greencover/weights/attunetgreen_128_1class_binary_extenddata3.h5')
        args_parser.add_argument('--weight_path_water', help='Path of water detect model weights', required=False, 
                                type=str, default='/home/quyet/WorkSpace/Quyet/greencover/weights/unet3pluswater_256_1class_binary_val.h5')
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
        folder_paths = '/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/T10_raw/test'
        weight_path_cloud = '/home/quyet/WorkSpace/Quyet/greencover/weights/cloud_weights.h5'
        weight_path_green = '/home/quyet/WorkSpace/Quyet/greencover/weights/green_weights.h5'
        weight_path_water = '/home/quyet/WorkSpace/Quyet/greencover/weights/water_weights.h5'

    box_path = os.path.join(folder_paths, 'box')
    tmp_path = os.path.join(folder_paths, 'tmp')
    base_path = os.path.join(folder_paths,'base')
    results_path = os.path.join(folder_paths,'results')
    list_month = sorted(glob.glob(os.path.join(folder_paths,'T*')))
    namee = os.path.basename(list_month[0])
    crs = rasterio.open(glob.glob(os.path.join(folder_paths, namee,'*.tif'))[0]).crs.to_string()
    list_removecloud_img = []

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    print("*Start check base image...")
    if glob.glob(os.path.join(base_path,'*.tif')):
        exist_base = True
        base_image = glob.glob(os.path.join(base_path,'*.tif'))[0]
        print("**Base image is exists")
    else:
        base_image = None

    check_month=[]
    for i in list_month:
        name = os.path.basename(i)
        check_month.append(os.path.basename(i))
        folder_path = os.path.join(folder_paths, 'results', name)
        if glob.glob(os.path.join(i,'*.tif')) == []:
            pass
        else:
            print("*Start check tmp and crop image for %s..."%(name))
            if os.path.exists(os.path.join(tmp_path,name+'_cut')):
                out_dir = os.path.join(tmp_path,name+'_cut')
                print("**Exist cut image folder %s"%(name+'_cut'))
                pass
            else:
                print("***Check result of cloud remove:")
                if not os.path.exists(os.path.join(folder_path,name+'.tif')):
                    print("****", 'Not exist')
                    print("***Crop image %s..."%(name))
                    out_dir = step_1(i, box_path, tmp_path)
                else:
                    print("****", 'Exist')       
    
            print("*Start predict cloud and remove it for %s..."%(name))
            if not os.path.exists(os.path.join(folder_path,name+'.tif')):
                print("**Cloud detection %s..."%(os.path.basename(i)))
                print("**Check results old predict cloud:")
                if not os.path.exists(os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud')):
                    print("***Not exist")
                    out_img_cloud = step_2(out_dir, tmp_path, weight_path_cloud, base_image)
                else:
                    num_img = len(glob.glob(os.path.join(out_dir,'*.tif')))
                    num_predict = len(glob.glob(os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud','*.tif')))
                    if num_img == num_predict:
                        print("***Exist")
                        out_img_cloud = os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud')
                    else:
                        out_img_cloud = step_2(out_dir, tmp_path, weight_path_cloud, base_image)
            else:
                out_img_cloud = os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud')
                print("**Exist cloud remove result %s"%(name))
                pass
            
            print("*Create folder results")
            new_folder = os.path.join(folder_path, name+'.tif')
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            print("*Merge image and mosaic with PCI...")
            # if the result of mosaic is exist, so pass mosaic
            if not os.path.exists(cloud_tmp_dir):
                os.mkdir(cloud_tmp_dir)

            cloud_tmp_dirr = os.path.join(cloud_tmp_dir, name)
            # if not os.path.exists(os.path.join(folder_path, name+'.tif')):
            #     out_merge_mosaic = merge_mosaic(out_img_cloud, cloud_tmp_dirr, temp_folder, base_path, name)
            #     shutil.copyfile(out_merge_mosaic, new_folder)
            # else:
            #     out_merge_mosaic = os.path.join(temp_folder, name+'.tif')

            # To do:
            if not os.path.exists(os.path.join(folder_path,name+'.tif')):
                mosaic_img_tmp = out_img_cloud = os.path.join(tmp_path, name, name+'.tif')
                shutil.copyfile(mosaic_img_tmp, new_folder)

            print("******", os.path.basename(name),"******",)
            list_removecloud_img.append(new_folder)

    # if not exist_base:
    #     try:
    #         list_img_error = step_3(list_removecloud_img, base_path, crs)
    #         print("list_img_error",list_img_error)
    #         for i in list_img_error:
    #             os.remove(os.path.join(results_path,i,i+'.tif'))
    #             out_merge_mosaic = merge_mosaic(out_img_cloud, cloud_tmp_dirr, temp_folder, base_path, i)
    #             shutil.copyfile(out_merge_mosaic, new_folder)
    #     except:
    #         raise Exception("Error with base file, Don't have any base file.")

    for j in sorted(list_removecloud_img):
        print("******Classification greencover %s******"%(os.path.basename(j).split('.')[0]))
        new_folder_path = os.path.join(folder_paths, 'results', os.path.basename(j).split('.')[0])
        result_green, result_water = step_4(j, new_folder_path, weight_path_green, weight_path_water)
        print("*Colormap results")
        result_all = combine_all(j, result_green, result_water)

        # month = os.path.basename(j).split('.')[0]
        # img_tmp = os.path.join(folder_paths, 'tmp', month, os.path.basename(j))
        # name_img = os.path.basename(j).replace('.tif', '_DA.tif')
        # folder_mosaic = os.path.join(new_folder_path, 'DA')
        # if os.path.exists(folder_mosaic):
        #     os.mkdir(folder_mosaic)
        # img_mosaic = os.path.join(folder_mosaic, name_img)
        # shutil.copyfile(img_tmp, img_mosaic)
        # result_green_tmp, result_water_tmp = step_4(img_mosaic, folder_mosaic, weight_path_green, weight_path_water)
        # result_all = combine_all(img_mosaic, result_green_tmp, result_water_tmp)

    print("*Clean workspace")
    try:
        shutil.rmtree(temp_folder)
        shutil.rmtree(cloud_tmp_dir)
    except:
        raise Exception("Can't remove tmp folder, please check it.")
    shutil.rmtree(os.path.join(folder_paths,'tmp'))

    if static_result:
        threshold = 0.1
        check_result(results_path, check_month, threshold)
    print("******Finished******")
