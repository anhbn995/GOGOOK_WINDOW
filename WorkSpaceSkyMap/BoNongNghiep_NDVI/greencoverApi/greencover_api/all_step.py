from cloud_remove.mosaic import mosaic
import os
import glob
import shutil


from change_model import get_model
from utils import get_list_fp, renew_baseimg
from check_results import main as check_result
from greencover import detect_green, detect_water
from greencover.merge_water_green import combine_all
from mosaic.mosaic_image import main as merge_mosaic
from mosaic.check_img import pick_base, count_compare
from cloud_remove.cloud_remove import main as cloud_remove
from download_crop.crop_image import main_cut_img as crop_image
from workspace import init_workspace, init_input_fn, check_exists_foler

def step_1(img_path, box_path, tmp_path):
    out_dir = crop_image(img_path,box_path,tmp_path)
    return out_dir

def step_2(img_dir, tmp_dir, weight_path_cloud, base_image=None):
    name = os.path.basename(img_dir)
    name = name.split('_')[0]
    list_fp_img_selected = get_list_fp(img_dir)

    sort_amount_of_clouds = True
    first_image = None
    tmp_dir = os.path.join(tmp_dir, name)
    out_fp = os.path.join(tmp_dir, name + ".tif")

    FN_MODEL = weight_path_cloud
    out_folder = cloud_remove(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image)
    return out_folder

def step_3(list_removecloud_img, base_path, crs):
    base_img_path = pick_base(list_removecloud_img, base_path)
    base_img_path_recrs = renew_baseimg(base_img_path, crs)
    os.remove(base_img_path)
    list_image_error = []
    for i in list_removecloud_img:
        print("list_removecloud_img",os.path.basename(i))
        status = count_compare(i, base_img_path_recrs)
        print(status)
        if not status:
            list_image_error.append(os.path.basename(i))
    return list_image_error

def step_4(img, result_path, weight_path_green, weight_path_water, dil=True):
    green_model, input_size_green = get_model("Green_model")
    water_model, input_size_water = get_model("Water_model")
    tmp_green = detect_green.predict(img, result_path, weight_path_green, green_model, input_size_green, dil=dil)
    # tmp_green = detect_green.predict(img, result_path, weight_path_green, dil=False)
    tmp_water = detect_water.predict(img, result_path, weight_path_water, water_model, input_size_water)
    # tmp_water = detect_water.predict(img, result_path, weight_path_water)
    return tmp_green, tmp_water

if __name__ == '__main__':
    debug = False
    # Doc thong tin tu file json de lay thong tin ve du lieu
    static_result, temp_dir, folder_paths, weight_path_cloud, weight_path_green, \
                                weight_path_water, box_aoi, list_month, crs = init_input_fn(debug)

    # Khoi tao cac thu muc de chua file 
    box_path, tmp_path, base_path, results_path, temp_folder, cloud_tmp_dir = init_workspace(folder_paths, temp_dir, box_aoi, crs)
    
    check_month=[]
    list_removecloud_img = []
    print("*Start check base image...")
    if glob.glob(os.path.join(base_path,'*.tif')):
        base_image = glob.glob(os.path.join(base_path,'*.tif'))[0]
        print("**Base image is exists")
    else:
        base_image = None
        print("**Base image isn't exists")

    for i in list_month:
        name = os.path.basename(i)
        check_month.append(os.path.basename(i))
        folder_path = os.path.join(folder_paths, 'results', name)
        check_exists_foler(folder_path)

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

            print("*Merge image and mosaic with PCI...")
            check_exists_foler(cloud_tmp_dir)
            cloud_tmp_dirr = os.path.join(cloud_tmp_dir, name)
            if not os.path.exists(os.path.join(folder_path, name+'.tif')):
                out_merge_mosaic = merge_mosaic(out_img_cloud, cloud_tmp_dirr, temp_folder, base_path, name)
                shutil.copyfile(out_merge_mosaic, new_folder)
            else:
                out_merge_mosaic = os.path.join(temp_folder, name+'.tif')

            print("******", os.path.basename(name),"******",)
            list_removecloud_img.append(new_folder)

    for j in list_removecloud_img:
        print("******Classification greencover %s******"%(os.path.basename(j).split('.')[0]))
        new_folder_path = os.path.join(folder_paths, 'results', os.path.basename(j).split('.')[0])
        result_green, result_water = step_4(j, new_folder_path, weight_path_green, weight_path_water)
        print("*Colormap results")
        result_all = combine_all(j, result_green, result_water)
        
        month = os.path.basename(j).split('.')[0]
        img_tmp = os.path.join(folder_paths, 'tmp', month, os.path.basename(j))
        img_mosaic = os.path.join(new_folder_path, 'DA', os.path.basename(j))
        folder_mosaic = os.path.join(new_folder_path, 'DA')
        check_exists_foler(folder_mosaic)
        if not os.path.exists(img_mosaic):
            shutil.copyfile(img_tmp, img_mosaic)
        result_green_tmp, result_water_tmp = step_4(img_mosaic, folder_mosaic, weight_path_green, weight_path_water)
        result_all = combine_all(img_mosaic, result_green_tmp, result_water_tmp)

    print("*Clean workspace")
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        if os.path.exists(cloud_tmp_dir):
            shutil.rmtree(cloud_tmp_dir)
    except:
        raise Exception("Can't remove tmp folder, please check it.")
    # shutil.rmtree(os.path.join(folder_paths,'tmp'))

    if static_result:
        threshold = 0.1
        check_result(results_path, check_month, threshold)
    print("******Finished******")
