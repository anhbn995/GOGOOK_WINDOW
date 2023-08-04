import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.merge import merge

def convert_obj(img):
    img1 = img/255
    img1[img1==1]=2
    img1[img1==0]=1
    img1[img1==2]=0
    return img1

def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=data.dtype,
                        crs=crs,
                        transform=tr,
                        nodata=0,
                        ) as dst:
                            dst.write(data)

def mosaic(list_img_name, out_path, base_img=None):
    src_files_to_mosaic = []
    # print(list_img_name)
    for name_f in list_img_name:
        fp = name_f
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    if base_img:
        src_files_to_mosaic.append(rasterio.open(base_img))
    mosaic, out_trans = merge(src_files_to_mosaic)
    write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)

list_img = glob.glob(os.path.join('/home/quyet/data/greencover_bangkok/tmp/T12/data_genorator_01/predict_float/cloud', '*.tif'))
if len(list_img) >1:
    out_path = list_img[0].replace('.tif','_abc.tif')
    mosaic(list_img, out_path)
else:
    out_path = list_img[0]
    print(out_path)

mask = rasterio.open(out_path).read_masks(1)
mask = convert_obj(mask)
plt.imshow(mask)

mask1 = rasterio.open('/home/quyet/data/greencover_bangkok/base/S2ฺB_11-01-2021_BKK copy.tif').read_masks(1)
mask1 = convert_obj(mask1)

mask2 = mask - mask1
mask2[mask2<0]=0
plt.imshow(mask2)

with rasterio.open(out_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
new_dataset = rasterio.open(out_path.replace('.tif', '_abc.tif'), 'w', driver='GTiff',
                            height = h, width = w,
                            count=1, dtype="uint8",
                            crs=crs,
                            transform=transform1,
                            compress='lzw')
new_dataset.write(mask2.astype(np.uint8),1)
new_dataset.close()