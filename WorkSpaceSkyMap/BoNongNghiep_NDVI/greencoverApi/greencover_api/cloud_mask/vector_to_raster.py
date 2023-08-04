import os
import glob
import rasterio
import rasterio.features
import geopandas as gp

def create_list_id(path,end_str):
    num_str = len(end_str)
    list_id = []
    os.chdir(path)
    for file in glob.glob("*{}".format(end_str)):
        list_id.append(file[:-num_str])
        # print(file[:-4])
    return list_id

def create_mask_by_shapefile2(shape_path, height, width, tr):
    # shape_file = os.path.expanduser(shape_path)
    list_shape = create_list_id(shape_path,'.shp')
    # list_shape = [shape_path]
    list_geometry = []
    for shape_id in list_shape: 
        shape_file = os.path.join(shape_path,shape_id+'.shp')
        # shape_file = shape_id
        print(shape_file)
        shp = gp.read_file(shape_file)
        ls_geo = [(x.geometry) for i, x in shp.iterrows()]
        list_geometry.extend(ls_geo)
    mask = rasterio.features.rasterize(list_geometry
                                    ,out_shape=(height, width)
                                    ,transform=tr)
    return mask

def arr2raster(path_out, bands, height, width, tr, dtype="uint8",crs=None,projstr=None):
    num_band = len(bands)
    # if coordinate!= None:
    #     crs = rasterio.crs.CRS.from_epsg(coordinate)
    # else:
    #     crs = rasterio.crs.CRS.from_string(projstr)
    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                            height = height, width = width,
                            count = num_band, dtype = dtype,
                            crs = crs,
                            transform = tr,
                            # nodata = 0,
                            compress='lzw')
    if num_band == 1:
        new_dataset.write(bands[0], 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i],i+1)
    new_dataset.close()

def build_mask2(image_id,img_dir,path_shape):
    path_image = os.path.join(img_dir,image_id+'.tif')
    # name_output = glob.glob(os.path.join(img_dir, '*.shp'))[0]
    output_mask =  os.path.join(img_dir,image_id +'_water.tif')

    with rasterio.open(path_image) as src:
        tr = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        print(projstr)
        crs = src.crs
        check_epsg = crs.is_epsg_code
        # if check_epsg:
        coordinate = src.crs.to_epsg()
        # else:
        # coordinate = None
    mask1 = create_mask_by_shapefile2(path_shape, h, w, tr)
    # mask1[mask1==0]=2
    # mask1[mask1==1]=0
    # mask1[mask1==2]=1
    # mask2 = cv2.bitwise_not(mask1)
    print(output_mask)
    arr2raster(output_mask, [mask1], h, w, tr, dtype="uint8",crs=crs,projstr=projstr)

if __name__ == "__main__":
    img_dir = '/home/quyet/data/bk_2/results/shp/T2'
    image_id = sorted(glob.glob(os.path.join(img_dir, '*.tif')))[0].split('/')[-1].replace('.tif', '')
    path_create = img_dir
    path_shape = img_dir
    build_mask2(image_id,img_dir,path_shape)