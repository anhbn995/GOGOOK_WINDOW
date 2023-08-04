from genericpath import exists
import os
import glob
import shutil
from cloud_remove.cloud_remove import main

if __name__ == "__main__":
    workspace = '/home/quyet/data/GEOMIN/GEOMINA'
    results = os.path.join(workspace, 'results')
    FN_MODEL = '/home/quyet/data/20211213_124156_0412_val_weights.h5'
    sort_amount_of_clouds = True
    first_image = None
    tmp_path = os.path.join(workspace, 'tmp')
    list_folder = os.listdir(workspace)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    if not os.path.exists(results):
        os.mkdir(results)

    # for i in list_folder:
    #     folder_path = os.path.join(workspace, i)
    #     list_month = glob.glob(os.path.join(folder_path, 'T*'))
    #     abc = os.path.join(tmp_path, i)
    #     if not os.path.exists(abc):
    #         os.mkdir(abc)
    #     print(list_month)
    #     for j in list_month:
    #         list_fp_img_selected = [sorted(glob.glob(os.path.join(j, '*.tif')))[0]]
    #         print(list_fp_img_selected)
    #         name = i + '_' + os.path.basename(j)
    #         tmp_dir = os.path.join(tmp_path, name)
    #         if not os.path.exists(tmp_dir):
    #             os.mkdir(tmp_dir)
    #         out_fp = os.path.join(tmp_dir, name + ".tif")
    #         main(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image=None)
    #         in_path = os.path.join(tmp_dir, 'data_genorator_01/predict_float', os.path.basename(list_fp_img_selected[0]))
    #         out_path = os.path.join(abc, os.path.basename(list_fp_img_selected[0]).replace('.tif', '_cloud.tif'))
    #         shutil.copyfile(in_path, out_path)

    for i in list_folder:
        if i == 'results':
            pass
        elif i == 'tmp':
            pass
        else:
            folder_path = os.path.join(workspace, i)
            print(folder_path)
            list_fp_img_selected = glob.glob(os.path.join(folder_path, '*.tif'))
            tmp_dir = os.path.join(tmp_path, i)
            out_fp = os.path.join(tmp_dir, i + ".tif")
            if not os.path.exists(os.path.join(results, os.path.basename(list_fp_img_selected[0]).replace('.tif', '_cloud.tif'))):
                main(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image=None)
                list_cloud_mask = glob.glob(os.path.join(tmp_dir, 'data_genorator_01/predict_float', '*.tif'))
                for j in list_cloud_mask:
                    shutil.move(j, os.path.join(results, os.path.basename(j).replace('.tif', '_cloud.tif')))