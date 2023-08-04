import os
import glob
import shutil

from run_v2 import run_segmentation
from greencover.merge_water_green import combine_all

def main(input_path, weight_path_green, weight_path_water, dil):
    # list_month = glob.glob(os.path.join(input_path, 'T*'))
    # for j in list_month:
    #     if j.split('.')[-1] == 'tif':
    #         list_month = None

    # if list_month:
    #     for i in list_month:
    #         img = glob.glob(os.path.join(i, '*.tif'))[0]
    #         # result_green, result_water = run_segmentation(img, i, weight_path_green, weight_path_water, dil)
    #         result_green = run_segmentation(img, i, weight_path_green, weight_path_water, dil)
    #         # combine_all(img, result_green, result_water)
    # else:
    list_folder = os.listdir(input_path)
    for j in list_folder:
        folder_path = os.path.join(input_path, j)
        list_img = glob.glob(os.path.join(folder_path, '*.tif'))
        for i in list_img:
            if '_green.tif' in i:
                pass
            elif '_water.tif' in i:
                pass
            else:
                # out_path = os.path.join(input_path, os.path.basename(i).replace('.tif', ''))
                # if not os.path.exists(out_path):
                #     os.mkdir(out_path)
                result_green, result_water = run_segmentation(i, folder_path, weight_path_green, weight_path_water, dil)
                # result_green = run_segmentation(i, input_path, weight_path_green, weight_path_water, dil)
                # combine_all(i, result_green, result_water)    
    return True

if __name__=="__main__":
    input_path = '/home/quyet/data/GEOMIN/data_extend_for_water/abc'
    weight_path_green = '/home/quyet/WorkSpace/Model/Segmen_model/weights/attunetgreen_128_1class_new_planet_val.h5'
    # weight_path_water = '/home/quyet/WorkSpace/Model/Segmen_model/weights/attunetwater_128_1class_new_planet_val.h5'
    weight_path_water = '/home/quyet/WorkSpace/Model/Segmen_model/weights/unet3plus_water_256_1class_binary_geomin_train.h5'
    dil=False
    main(input_path, weight_path_green, weight_path_water, dil=dil)
    print("Finished")