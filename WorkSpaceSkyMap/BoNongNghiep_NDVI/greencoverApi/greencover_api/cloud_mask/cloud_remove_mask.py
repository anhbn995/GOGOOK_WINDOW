import os
import glob
from re import A
import fiona
import rasterio
import geopandas
import numpy as np
from cloud_mask.raster_to_vector import raster_to_vector, raster_to_vector_2

def intersection_cloud_2(tif_path, outputFileName):
    list_tif = glob.glob(os.path.join(tif_path, '*.tif'))
    for i in list_tif:
        img = rasterio.open(list_tif[0]).read()[0]
        if  len(img[img != 0]) == 0:
            check_status = False
        else:
            check_status =True
    if check_status:
        img = rasterio.open(list_tif[0]).read()[0]/255
        for i in list_tif[1:]:
            img +=(rasterio.open(i).read()[0]/255)

        max_number = np.max(img)
        img[img!=max_number]=0
        img[img==max_number]=1
    base_path = list_tif[0]
    raster_to_vector_2(base_path, img ,outputFileName)

def intersection_cloud(tif_path, tmp_path, out_path):
    list_tif = glob.glob(os.path.join(tif_path, '*.tif'))
    list_true = []
    for i in list_tif:
        img = rasterio.open(i).read()[0]
        if len(img[img != 0]) == 0:
            check_status = False
            break
        else:
            list_true.append(i)
            check_status =True
            
    if check_status:
        shp_path = convert_mask_to_shp(list_true, tmp_path)
        list_shp = glob.glob(os.path.join(shp_path, '*.shp'))
        res_intersection = geopandas.read_file(list_shp[0])
        
        print("Create interection cloud...")
        for i in list_shp[1:]:
            df = geopandas.read_file(i)
            # if len(df.geometry) == 0:
            #     pass

            print(res_intersection)
            print(df)
            res_intersection = geopandas.overlay(res_intersection, df, how='intersection')
            # else: 
            #     from shapely.geometry import Polygon, mapping
            #     schema = {
            #         'geometry':'Polygon',
            #         'properties':{'FID':'str'}
            #     }
            #     with fiona.open(out_path, "w", driver='ESRI Shapefile', schema=schema) as source:
            #         source.write({'geometry': mapping(Polygon([])),
            #                         'properties': {'FID':'empty'}
            #                     })
            #     break

        if len(res_intersection.geometry) == 0:
            from shapely.geometry import Polygon, mapping
            schema = {
                'geometry':'Polygon',
                'properties':{'FID':'str'}
            }
            with fiona.open(out_path, "w", driver='ESRI Shapefile', schema=schema) as source:
                source.write({'geometry': mapping(Polygon([])),
                                'properties': {'FID':'empty'}
                            })
        else:
            res_intersection.to_file(out_path)
    else:
        from shapely.geometry import Polygon, mapping
        schema = {
            'geometry':'Polygon',
            'properties':{'FID':'str'}
        }
        with fiona.open(out_path, "w", driver='ESRI Shapefile', schema=schema) as source:
            source.write({'geometry': mapping(Polygon([])),
                            'properties': {'FID':'empty'}
                        })
    print("Finished")
    
def convert_mask_to_shp(list_tif, tmp_path):
    # list_tif = glob.glob(os.path.join(tif_path, '*.tif'))
    result_folder = os.path.join(tmp_path, 'cloud_shp')
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    for i in list_tif:
        name_shp = os.path.join(result_folder, os.path.basename(i).replace('.tif', '.shp'))
        raster_to_vector(i, name_shp)
    return result_folder

if __name__=="main":
    tif_path = ''
    outputFileName = '' 
    intersection_cloud_2(tif_path, outputFileName) 