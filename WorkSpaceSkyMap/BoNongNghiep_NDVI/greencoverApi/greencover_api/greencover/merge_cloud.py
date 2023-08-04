import os
import glob
from numpy.lib.type_check import imag
import rasterio
import numpy as np

def combine_cloud(color_path, cloud_path):
    with rasterio.open(color_path) as src:
        color_img = src.read()[0]
        out_meta = src.meta

    if cloud_path:
        with rasterio.open(cloud_path) as src1:
            cloud_img = (src1.read()[0]/255) *4

        results_img = color_img + cloud_img
        results_img[results_img==7] = 4
        results_img[results_img==6] = 4
        results_img[results_img==5] = 1
    else:
        results_img = color_img

    combine_path = color_path.replace('.tif','_combine_cloud.tif')
    with rasterio.Env():
        profile = out_meta
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
        with rasterio.open(combine_path, 'w', **profile) as dst:
            dst.write(results_img.astype(np.uint8),1)

    result_color = color_path.replace('.tif','_color_final.tif')
    with rasterio.Env():   
        with rasterio.open(combine_path) as src:  
            shade = src.read()[0]
            meta = src.meta.copy()
            meta.update({'nodata': 0, 'dtype':'uint8'})
        with rasterio.open(result_color, 'w', **meta) as dst:
            dst.write(shade, indexes=1)
            dst.write_colormap(1, {
                    0: (0, 0, 0, 0),
                    1: (34, 139, 34, 0), #Green1
                    # 2: (157,193,131,0), #Green2
                    2: (100, 149, 237, 0), #water
                    3: (101, 67, 33, 0), 
                    4: (72, 72, 72, 0)
                    }) #Buildup

if __name__ == "__main__":
    workspace = '/home/quyet/data/GEOMIN/8bit_perimage/results_v2_24122021_add_cloud'
    list_folder = os.listdir(workspace)
    # for i in list_folder:
    #     foler_path = os.path.join(workspace, i)
    #     list_image = sorted(glob.glob(os.path.join(foler_path, '*.tif')))
    #     base_name = list_image[0]
    #     color_path = base_name.replace('.tif', '_color.tif')
    #     if os.path.exists(base_name.replace('.tif', '_cloud.tif')):
    #         cloud_path = base_name.replace('.tif', '_cloud.tif')
    #     else:
    #         cloud_path = None
        
    #     print("***",cloud_path)
    #     combine_cloud(color_path, cloud_path)

    for j in list_folder:
        aaa = os.path.join(workspace, j)
        bbb = os.listdir(aaa)
        for i in bbb:
            foler_path = os.path.join(aaa, i)
            list_image = sorted(glob.glob(os.path.join(foler_path, '*.tif')))
            base_name = list_image[0]
            color_path = base_name.replace('.tif', '_color.tif')
            if os.path.exists(base_name.replace('.tif', '_cloud.tif')):
                cloud_path = base_name.replace('.tif', '_cloud.tif')
            else:
                cloud_path = None
            if os.path.exists(base_name.replace('.tif', '_color_color_final.tif')):
                pass
            else:
                print("***",cloud_path)
                combine_cloud(color_path, cloud_path)