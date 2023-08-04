import os
import glob
from tqdm import tqdm

from cloud_remove.mosaic import main_mosaic
from cloud_remove.gendata import main_gendata
from cloud_remove.predict_cloud import main_predict_cloud


def main(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image=None):
    dir_img_float_tmp = main_gendata(list_fp_img_selected, tmp_dir)
    # print("---",dir_img_float_tmp)
    dir_predict_cloud_tmp = main_predict_cloud(dir_img_float_tmp, FN_MODEL)
    # print("abc",dir_predict_cloud_tmp)
    # dir_predict_cloud_tmp = r"/home/skm/SKM/WORK/Cloud_and_mosaic/INDONESIA_SAMARINDA/img_origin_cut_img/tmpv4/data_genorator_01/predict_float"
    aaa = main_mosaic(list_fp_img_selected, dir_predict_cloud_tmp, out_fp, sort_amount_of_clouds, first_image, base_image)
    return aaa


if __name__ == "__main__":
    workspace = '/home/quyet/data/aaaaa/'
    FN_MODEL = '/home/quyet/data/20211210_163952_0167_val_weights.h5'
    sort_amount_of_clouds = True
    first_image = None
    tmp_path = os.path.join(workspace, 'tmp')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    list_folder = os.listdir(workspace)

    for i in list_folder:
        folder_path = os.path.join(workspace, i)
        list_month = glob.glob(os.path.join(folder_path, 'T*'))
        for j in list_month:
            list_fp_img_selected = glob.glob(j, '*.tif')
            name = i + '_' + os.path.basename(j)
            tmp_dir = os.path.join(tmp_path, name)
            out_fp = os.path.join(tmp_dir, name + ".tif")
            main(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image=None)
    