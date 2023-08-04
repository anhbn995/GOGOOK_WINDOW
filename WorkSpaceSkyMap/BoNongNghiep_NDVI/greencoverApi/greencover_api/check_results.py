import os
import rasterio
import numpy as np

def check_green_month(result_path, list_month, threshold=0.1, type_img=None):
    cur_green = 0
    list_error_results = []
    min_t = 1.0 - threshold
    max_t = 1.0 + threshold
    for i in list_month:
        name = i
        if type_img:
            result_path_img = os.path.join(result_path,name, type_img,name+'_'+type_img+'_color.tif')
        else:
            result_path_img = os.path.join(result_path,name,name+'_color.tif')
        mask = rasterio.open(result_path_img).read()
        cur_pixel = np.sum(mask[mask==1])
        if cur_green == 0:
            cur_green = cur_pixel
            pass
        else:
            print(name, cur_pixel)
            print(cur_green)
            if cur_pixel < min_t*cur_green:
                percent = ((cur_green - cur_pixel)/cur_green)*100
                list_error_results.append([result_path_img, 'giam', percent])
            elif cur_pixel > max_t*cur_green:
                percent = ((cur_pixel - cur_green)/cur_green)*100
                list_error_results.append([result_path_img, 'tang', percent])
            cur_green = cur_pixel
    return list_error_results

def main(result_path, list_month, threshold=0.1, type_img=None):
    list_error_results = check_green_month(result_path, list_month, threshold, type_img)
    if list_error_results:
        for i in list_error_results:
            print("Can kiem tra %s, %s so vs thang truoc %s%%"%(i[0],i[1], str(i[2])))
    else:
        print("The results should not differ by more than %s%%."%str(threshold*100))

if __name__=="__main__":
    # result_path = '/home/nghipham/Desktop/Jupyter/data/DA/7_GreenCover_skymap/Green Cover Bangkok Thailand/results'
    result_path = '/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing/Green Cover Npark Singapore/results'
    # list_month = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']
    list_month = ['T10','T11']
    threshold = 0.05
    print("Check results PCI...")
    type_img = None
    main(result_path, list_month, threshold=threshold, type_img=type_img)
    # print("Check results GDAL...")
    # type_img = 'DA'
    # main(result_path, list_month, threshold=threshold, type_img=type_img)